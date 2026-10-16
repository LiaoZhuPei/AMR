"""
纯 TCN：直接处理 (batch, 1024, 2)
Conv1d(2→channels) → TCN Blocks(膨胀卷积 ×10层) → GAP → Classifier

感受野计算：
  kernel=3, dilations=[1,2,4,8,16,32,64,128,256,512]
  每层单侧扩展 = dilation
  总感受野 = 1 + 2 × Σ(dilations) = 1 + 2×1023 = 2047 > 1024 ✓
  → 10 层即可覆盖整个序列
"""
import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """TCN 残差块：两层膨胀卷积 + 残差"""

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # 对称 padding，保持长度不变

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class PureTCN(nn.Module):
    """
    输入: (batch, 1024, 2)
    无 CNN，用一层普通卷积将通道 2→128，然后 10 层 TCN
    """

    def __init__(self, num_classes=9, channels=128, num_layers=10,
                 kernel_size=3, dropout=0.2):
        super().__init__()

        # 通道扩展：2 → channels（不改变序列长度）
        self.input_conv = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        # TCN 主体：10 层膨胀卷积
        tcn_blocks = []
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
            tcn_blocks.append(TCNBlock(channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_blocks)

        self.final_norm = nn.BatchNorm1d(channels)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
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
        # x: (batch, 1024, 2)
        x = x.permute(0, 2, 1)    # → (batch, 2, 1024)
        x = self.input_conv(x)     # → (batch, 128, 1024)  长度不变！
        x = self.tcn(x)            # → (batch, 128, 1024)  长度不变！
        x = self.final_norm(x)
        x = x.mean(dim=2)         # GAP → (batch, 128)
        return self.classifier(x)


def build_model(classes=9):
    return PureTCN(num_classes=classes)


if __name__ == '__main__':
    model = build_model(9)
    x = torch.randn(4, 512, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")

    # 验证感受野
    print(f"\nReceptive field: 1 + 2*sum([2^i for i in range(10)]) = "
          f"{1 + 2 * sum([2 ** i for i in range(10)])}")