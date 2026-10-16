"""
model_bilstm_v2.py
双分支: CNN→BiLSTM→Attention (时序) + HigherOrderStats (统计) → 融合
升级点:
  1. 使用多尺度CNN前端 (cnn_frontend_v2)
  2. 新增统计特征分支 (42维→96维)
  3. 更大的融合分类器
"""
import torch
import torch.nn as nn
from cnn_frontend_v2 import CNNFrontend
from stat_features import HigherOrderStats


class Attention(nn.Module):
    """时间维度自注意力：学习每个时间步的重要性权重"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: (B, seq_len, hidden)
        attn_weights = self.attention(lstm_output)            # (B, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights, dim=1)  # (B, hidden)
        return context


class DualBranchBiLSTM(nn.Module):
    """
    分支1: CNN(8x下采样) → BiLSTM → Attention → 256维
    分支2: HigherOrderStats(42维 → 96维)
    融合:  Cat(256+96=352) → 384 → 192 → num_classes

    对比原版改动:
      - CNN前端: 单尺度3层conv → 多尺度(3/7/15)+SE+残差
      - 新增统计分支: 直方图+Gini+峰度等42维特征
      - 分类器: 更深更宽
    """

    def __init__(self, num_classes=14, cnn_channels=128,
                 hidden_size=128, num_layers=2, dropout=0.3, stat_dim=96):
        super().__init__()

        # ── 分支1: 时序特征 ──

        # 多尺度CNN前端: (B, 2, L) → (B, cnn_channels, L//8)
        self.cnn = CNNFrontend(in_channels=2, out_channels=cnn_channels)

        # BiLSTM: (B, L//8, cnn_channels) → (B, L//8, hidden*2)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Attention: (B, L//8, hidden*2) → (B, hidden*2)
        self.attention = Attention(hidden_size * 2)

        # ── 分支2: 统计特征 ──
        self.stats = HigherOrderStats(proj_dim=stat_dim)

        # ── 融合分类器 ──
        temporal_dim = hidden_size * 2  # 256 (双向)
        fused_dim = temporal_dim + stat_dim  # 256 + 96 = 352
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(192, num_classes),
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
        # x: (B, L, 2)

        # 分支2: 统计特征（从原始 IQ 提取）
        stat_feat = self.stats(x)              # (B, 96)

        # 分支1: 时序特征
        h = x.permute(0, 2, 1)                # (B, 2, L)
        h = self.cnn(h)                        # (B, 128, L//8)
        h = h.permute(0, 2, 1)                # (B, L//8, 128)

        # BiLSTM
        lstm_out, _ = self.lstm(h)             # (B, L//8, 256)
        lstm_out = self.layer_norm(lstm_out)

        # Attention 加权聚合
        temporal_feat = self.attention(lstm_out)  # (B, 256)

        # 融合
        out = torch.cat([temporal_feat, stat_feat], dim=1)  # (B, 352)
        return self.classifier(out)


def build_model(classes=14):
    return DualBranchBiLSTM(num_classes=classes)


if __name__ == '__main__':
    model = build_model(14)
    print(model)

    # 测试 2048
    x = torch.randn(4, 2048, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # 测试 1024
    x2 = torch.randn(4, 1024, 2)
    out2 = model(x2)
    print(f"Input: {x2.shape} → Output: {out2.shape}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")