"""
model_cnn.py  (v2 — 抗过拟合版)
================================
针对 train≈0.9 / val≈0.7 过拟合问题的系统性修复：

  1. 缩减模型容量：3块卷积，通道 8→64→128→128，去掉原来 256ch 的第4块
  2. 每个卷积块后增加 Dropout(p=block_dropout)，而不是只在分类头
  3. 首层加 InstanceNorm1d 对齐 8 通道的值域差异
  4. 分类头加大 Dropout(p=head_dropout)
  5. 提供 build_model(regularize='light'/'medium'/'heavy') 三档快速调节

输入: (B, N_CHANNELS=8, 1024)
输出: (B, num_classes)
参数量: ~120 K（medium），远小于原来 ~400 K
"""

import torch
import torch.nn as nn
from feature_extract import N_CHANNELS


# ══════════════════════════════════════════════════════════════════════════
# 卷积块（含块内 Dropout）
# ══════════════════════════════════════════════════════════════════════════

def _conv_block(in_ch: int, out_ch: int, kernel: int,
                stride: int = 2, dropout: float = 0.0) -> nn.Sequential:
    pad = kernel // 2
    layers = [
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                  stride=stride, padding=pad, bias=False),
        nn.BatchNorm1d(out_ch),
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
    抗过拟合版 CNN。

    修改点（相对 v1 SimpleCNN）：
      · InstanceNorm1d 作为输入归一化，消除通道间值域差异
      · 通道上限缩至 128（原为 256），去掉第4卷积块
      · 每个卷积块末尾 Dropout(block_dropout)
      · 分类头 Dropout 加大至 head_dropout
      · kernel 序列改为 7→5→3，感受野不变但参数更少
    """

    def __init__(self,
                 num_classes: int = 14,
                 in_channels: int = N_CHANNELS,
                 block_dropout: float = 0.2,
                 head_dropout: float = 0.4):
        super().__init__()

        # ── 输入归一化（per-channel, per-sample，不引入跨样本统计）
        self.input_norm = nn.InstanceNorm1d(in_channels, affine=True)

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
        x = self.input_norm(x)    # per-channel instance norm
        x = self.encoder(x)       # (B, 128, 128)
        x = x.mean(dim=2)         # GAP → (B, 128)
        return self.head(x)       # (B, num_classes)


# ══════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════

_PRESETS = {
    # (block_dropout, head_dropout)
    'light':  (0.1, 0.3),   # 欠拟合时尝试
    'medium': (0.2, 0.4),   # 默认，针对当前 train=0.9/val=0.7 设计
    'heavy':  (0.3, 0.5),   # 若 medium 仍过拟合时尝试
}


def build_model(classes: int = 14,
                in_channels: int = N_CHANNELS,
                regularize: str = 'medium') -> RegularizedCNN:
    """
    regularize: 'light' | 'medium' | 'heavy'
      · light  — 过拟合轻微时使用（train-val 差距 <0.05）
      · medium — 当前场景默认（train≈0.9, val≈0.7）
      · heavy  — 仍然严重过拟合时使用
    """
    bd, hd = _PRESETS[regularize]
    return RegularizedCNN(
        num_classes=classes,
        in_channels=in_channels,
        block_dropout=bd,
        head_dropout=hd,
    )


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    for preset in ('light', 'medium', 'heavy'):
        m = build_model(14, regularize=preset)
        x = torch.randn(4, N_CHANNELS, 1024)
        out = m(x)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{preset:6s}]  output={out.shape}  params={params:,}")
