"""
model_cnn.py
============
轻量 CNN 分类器，输入为 8 通道特征序列 (N, N_CHANNELS, L)。

架构：3 个卷积块（逐步降采样 + 通道扩张）→ GAP → FC 分类头
不使用 TCN/Transformer/Mamba，无残差，只有纯卷积+BN+激活。

  输入: (B, 8, 1024)
  Block1: Conv(8→64,  k=7, s=2) → BN → GELU → (B, 64,  512)
  Block2: Conv(64→128, k=5, s=2) → BN → GELU → (B, 128, 256)
  Block3: Conv(128→256,k=3, s=2) → BN → GELU → (B, 256, 128)
  Block4: Conv(256→256,k=3, s=2) → BN → GELU → (B, 256,  64)
  GAP:   mean(dim=2)             → (B, 256)
  Head:  Dropout → Linear(256 → num_classes)

参数量约 ~400 K，训练快，推理轻。
"""

import torch
import torch.nn as nn
from feature_extract import N_CHANNELS


def _conv_block(in_ch: int, out_ch: int, kernel: int,
                stride: int = 2, dropout: float = 0.0) -> nn.Sequential:
    pad = kernel // 2
    layers = [
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                  stride=stride, padding=pad, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.GELU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    """
    8 通道特征 → 多层 1-D CNN → GAP → 全连接分类

    forward 输入 shape: (B, N_CHANNELS, L)
    """

    def __init__(self, num_classes: int = 14,
                 in_channels: int = N_CHANNELS,
                 dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            _conv_block(in_channels,  64,  kernel=7, stride=2),   # L/2
            _conv_block(64,          128,  kernel=5, stride=2),   # L/4
            _conv_block(128,         256,  kernel=3, stride=2),   # L/8
            _conv_block(256,         256,  kernel=3, stride=2),   # L/16
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
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
        x = self.encoder(x)     # (B, 256, L/16)
        x = x.mean(dim=2)       # GAP → (B, 256)
        return self.head(x)     # (B, num_classes)


def build_model(classes: int = 14,
                in_channels: int = N_CHANNELS,
                dropout: float = 0.3) -> SimpleCNN:
    return SimpleCNN(num_classes=classes,
                     in_channels=in_channels,
                     dropout=dropout)


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    model = build_model(14)
    x = torch.randn(4, N_CHANNELS, 1024)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
