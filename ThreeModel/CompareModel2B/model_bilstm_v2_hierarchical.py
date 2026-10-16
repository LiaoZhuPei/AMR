"""
model_bilstm_v2_hierarchical.py
两阶段层次识别双头模型
  Stage 1 (粗分类): 判断信号是否属于 QAM 族 → (QAM族 / 非QAM)
  Stage 2 (细分类): 在 QAM 族内判断具体阶数 → (16QAM / 32QAM / 64QAM / 128QAM / 256QAM)

架构:
  共享主干: CNN(多尺度) + BiLSTM + Attention + 统计特征
  分叉两个独立分类头:
    head_coarse: 2类  (0=非QAM, 1=QAM族)
    head_fine  : 5类  (QAM 内部细分)

训练: 联合损失 = L_coarse + λ * L_fine（仅 QAM 样本参与 fine loss）
推理: 先看 head_coarse，若预测为 QAM 再用 head_fine 输出具体阶数
"""

import torch
import torch.nn as nn
from cnn_frontend_v2 import CNNFrontend
from stat_features import HigherOrderStats

# ─────────────────────────────────────────────
# 类别定义（与训练脚本保持一致，顺序不要改）
# ─────────────────────────────────────────────
ALL_CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
               '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

QAM_CLASSES = ['16QAM', '32QAM', '64QAM', '128QAM', '256QAM']  # fine 头的顺序

# 工具函数：将全类别标签转换为粗/细标签
def make_coarse_label(label_idx, classes):
    """全类别索引 → 粗分类标签 (0=非QAM, 1=QAM)"""
    return 1 if classes[label_idx] in QAM_CLASSES else 0

def make_fine_label(label_idx, classes):
    """全类别索引 → 细分类标签 (0-4 in QAM_CLASSES, -1=非QAM)"""
    name = classes[label_idx]
    if name in QAM_CLASSES:
        return QAM_CLASSES.index(name)
    return -1  # 非 QAM，训练时会用 mask 过滤掉


# ─────────────────────────────────────────────
# Attention 模块
# ─────────────────────────────────────────────
class Attention(nn.Module):
    """时间维度自注意力"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (B, T, 1)
        return torch.sum(lstm_output * attn_weights, dim=1)               # (B, H)


# ─────────────────────────────────────────────
# 两阶段层次识别模型
# ─────────────────────────────────────────────
class HierarchicalBiLSTM(nn.Module):
    """
    共享主干 (CNN + BiLSTM + Attention + Stats) → 两个独立分类头

    head_coarse : Linear(fused_dim → 2)   粗分类: 非QAM(0) / QAM(1)
    head_fine   : Linear(fused_dim → 5)   细分类: 16/32/64/128/256QAM

    Args:
        num_all_classes : 全部类别数 (默认 14)
        num_qam_classes : QAM 内部类别数 (默认 5)
        cnn_channels    : CNN 输出通道数
        hidden_size     : BiLSTM 单向隐层大小
        num_layers      : LSTM 层数
        dropout         : Dropout 概率
        stat_dim        : 统计特征投影维度
    """

    def __init__(self, num_all_classes=14, num_qam_classes=5,
                 cnn_channels=128, hidden_size=128,
                 num_layers=2, dropout=0.3, stat_dim=96):
        super().__init__()
        self.num_qam_classes = num_qam_classes

        # ── 共享主干 ──────────────────────────────────────
        self.cnn        = CNNFrontend(in_channels=2, out_channels=cnn_channels)
        self.lstm       = nn.LSTM(
            input_size   = cnn_channels,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0,
            bidirectional= True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention  = Attention(hidden_size * 2)
        self.stats      = HigherOrderStats(proj_dim=stat_dim)

        temporal_dim = hidden_size * 2        # 256
        fused_dim    = temporal_dim + stat_dim  # 352

        # 共享特征进一步压缩（让两个头共用一个中间表示）
        self.shared_proj = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
        )

        # ── Stage 1: 粗分类头 (非QAM=0 / QAM=1) ─────────
        self.head_coarse = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 2),          # 2 类
        )

        # ── Stage 2: 细分类头 (QAM 内部 5 类) ────────────
        self.head_fine = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_qam_classes),   # 5 类
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, L, 2)  原始 IQ 输入

        Returns:
            logits_coarse : (B, 2)          粗分类 logits
            logits_fine   : (B, 5)          细分类 logits（对全部样本计算，训练时用 mask 过滤）
            shared_feat   : (B, 256)        共享特征（可选，用于可视化/分析）
        """
        # 统计特征分支
        stat_feat = self.stats(x)              # (B, 96)

        # 时序特征分支
        h = x.permute(0, 2, 1)                # (B, 2, L)
        h = self.cnn(h)                        # (B, 128, L//8)
        h = h.permute(0, 2, 1)                # (B, L//8, 128)
        lstm_out, _ = self.lstm(h)             # (B, L//8, 256)
        lstm_out = self.layer_norm(lstm_out)
        temporal_feat = self.attention(lstm_out)   # (B, 256)

        # 融合
        fused = torch.cat([temporal_feat, stat_feat], dim=1)   # (B, 352)
        shared_feat = self.shared_proj(fused)                  # (B, 256)

        # 两个头
        logits_coarse = self.head_coarse(shared_feat)   # (B, 2)
        logits_fine   = self.head_fine(shared_feat)     # (B, 5)

        return logits_coarse, logits_fine, shared_feat


def build_model(num_all_classes=14, num_qam_classes=5):
    return HierarchicalBiLSTM(
        num_all_classes = num_all_classes,
        num_qam_classes = num_qam_classes,
    )


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────
if __name__ == '__main__':
    model = build_model()
    print(model)

    x = torch.randn(4, 2048, 2)
    lc, lf, sf = model(x)
    print(f"Input : {x.shape}")
    print(f"Coarse: {lc.shape}   (2 classes: nonQAM / QAM)")
    print(f"Fine  : {lf.shape}   (5 classes: 16/32/64/128/256QAM)")
    print(f"Shared: {sf.shape}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
