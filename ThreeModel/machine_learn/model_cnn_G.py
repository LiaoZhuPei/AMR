"""
model_cnn.py  (v3 — 配合特征修正版)
==================================
配合 feature_extract.py (v3) 的修改版：
  1. 移除首层 InstanceNorm1d，防止其破坏特征原有的物理方差信息
  2. 保持 3块卷积，通道 8→64→128→128，有效控制模型容量
  3. 保持块内与分类头 Dropout，维持抗过拟合性能
"""

import torch
import torch.nn as nn
from feature_extract_G import N_CHANNELS


# ══════════════════════════════════════════════════════════════════════════
# 卷积块（含块内 Dropout）
# ══════════════════════════════════════════════════════════════════════════

def _conv_block(in_ch: int, out_ch: int, kernel: int,
                stride: int = 2, dropout: float = 0.0) -> nn.Sequential:
    pad = kernel // 2
    layers = [
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                  stride=stride, padding=pad, bias=False),
        nn.BatchNorm1d(out_ch),  # 批归一化在 batch 维度操作，保留了样本间的相对方差差异
        nn.GELU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════
# 主模型
# ══════════════════════════════════════════════════════════════════════════

class RegularizedCNN(nn.Module):
    """
    抗过拟合版 CNN (v3)。
    
    修改点：
      · 移除输入层 InstanceNorm1d，确保物理特征的方差分布（如幅度起伏）完好传递至卷积层
      · 保持轻量化架构，参数量约 120 K
    """

    def __init__(self,
                 num_classes: int = 14,
                 in_channels: int = N_CHANNELS,
                 block_dropout: float = 0.2,
                 head_dropout: float = 0.4):
        super().__init__()

        # ── 编码器：3块，1024→512→256→128
        self.encoder = nn.Sequential(
            _conv_block(in_channels,  64, kernel=7, stride=2,
                        dropout=block_dropout),   # 1024→512
            _conv_block(64,          128, kernel=5, stride=2,
                        dropout=block_dropout),   # 512→256
            _conv_block(128,         128, kernel=3, stride=2,
                        dropout=block_dropout),   # 256→128
        )

        # ── 分类头
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_CHANNELS, L)"""
        # 直接送入编码器，保留物理特征尺度
        x = self.encoder(x)       # (B, 128, 128)
        x = x.mean(dim=2)         # GAP → (B, 128)
        return self.head(x)       # (B, num_classes)


# ══════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════

_PRESETS = {
    'light':  (0.1, 0.3),
    'medium': (0.2, 0.4),
    'heavy':  (0.3, 0.5),
}


def build_model(classes: int = 14,
                in_channels: int = N_CHANNELS,
                regularize: str = 'medium') -> RegularizedCNN:
    bd, hd = _PRESETS[regularize]
    return RegularizedCNN(
        num_classes=classes,
        in_channels=in_channels,
        block_dropout=bd,
        head_dropout=hd,
    )


if __name__ == '__main__':
    m = build_model(14, regularize='medium')
    x = torch.randn(4, N_CHANNELS, 1024)
    out = m(x)
    print(f"Output shape: {out.shape}")