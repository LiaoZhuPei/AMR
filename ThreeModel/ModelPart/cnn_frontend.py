"""
共享 CNN 前端：所有模型使用完全相同的降采样模块
输入: (batch, 2, 1024) → 输出: (batch, 128, 128)
序列长度 8 倍压缩，特征维度 2→128
"""
import torch
import torch.nn as nn


class CNNFrontend(nn.Module):
    def __init__(self, in_channels=2, out_channels=128, input_length=512):
        super().__init__()
        if input_length >= 1024:
            # 三层降采样：1024 → 128
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(128, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_channels), nn.GELU(),
            )
        elif input_length >= 512:
            # 两层降采样：512 → 128
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
                nn.Conv1d(64, out_channels, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(out_channels), nn.GELU(),
            )
        else:
            # 一层降采样或不降采样
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(out_channels), nn.GELU(),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.cnn(x)


# 共享的分类头（所有模型都用同样的结构）
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


if __name__ == '__main__':
    cnn = CNNFrontend()
    x = torch.randn(4, 2, 1024)
    out = cnn(x)
    print(f"CNN Frontend: {x.shape} → {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in cnn.parameters()):,}")