"""
共享 CNN 前端：所有模型使用完全相同的降采样模块
输入: (batch, 2, 1024) → 输出: (batch, 128, 128)
序列长度 8 倍压缩，特征维度 2→128
"""
import torch
import torch.nn as nn


class CNNFrontend(nn.Module):
    def __init__(self, num_classes, in_channels=2, out_channels=128):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: (B, 2, 1024) → (B, 64, 512)
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            # Block 2: (B, 64, 512) → (B, 128, 256)
            nn.Conv1d(64, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),

            # Block 3: (B, 128, 256) → (B, 128, 128)
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    # 权重初始化
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # → (batch, 2, 1024)
        x = self.cnn(x)
        x = x.mean(dim=-1)
        x = self.head(x)
        return x

def build_model(classes=9):
    return CNNFrontend(num_classes=classes)

if __name__ == '__main__':
    cnn = CNNFrontend(num_classes = 2)
    x = torch.randn(4, 2, 1024)
    out = cnn(x)
    print(f"CNN Frontend: {x.shape} → {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in cnn.parameters()):,}")