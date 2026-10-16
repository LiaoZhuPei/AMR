"""
CNN + Temporal Convolutional Network (TCN)
核心：膨胀卷积，全并行，通过指数增长的 dilation 覆盖长距离依赖
"""
import torch
import torch.nn as nn
from cnn_frontend import CNNFrontend, ClassifierHead


class TCNBlock(nn.Module):
    """
    单个 TCN 残差块：两层膨胀卷积 + 残差连接
    使用对称 padding（非因果），适合分类任务
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # 对称 padding 使输出长度 = 输入长度
        padding = dilation * (kernel_size - 1) // 2

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
        # x: (batch, channels, seq_len)
        return x + self.net(x)  # 残差连接（通道数不变，无需投影）


class CNN_TCN(nn.Module):
    """
    架构: CNN(1024→128) → TCN(6层膨胀卷积) → GAP → Classifier
    输入: (batch, 1024, 2)

    感受野计算 (kernel=3, dilations=[1,2,4,8,16,32]):
      每层每侧扩展 dilation*(k-1)/2 = dilation
      总感受野 = 1 + 2*Σ(dilation_i) = 1 + 2*(1+2+4+8+16+32) = 127
      → 几乎覆盖整个 128 长度的序列 ✓
    """

    def __init__(self, num_classes=9, channels=128, num_layers=5,
                 kernel_size=3, dropout=0.2):
        super().__init__()

        self.cnn = CNNFrontend(in_channels=2, out_channels=channels)

        tcn_blocks = []
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32
            tcn_blocks.append(TCNBlock(channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_blocks)

        self.final_norm = nn.BatchNorm1d(channels)
        self.classifier = ClassifierHead(channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: (batch, 1024, 2)
        x = x.permute(0, 2, 1)       # → (batch, 2, 1024)
        x = self.cnn(x)               # → (batch, 128, 128)
        x = self.tcn(x)               # → (batch, 128, 128)
        x = self.final_norm(x)
        x = x.mean(dim=2)            # Global Average Pooling → (batch, 128)
        return self.classifier(x)


def build_model(classes=9):
    return CNN_TCN(num_classes=classes)


if __name__ == '__main__':
    model = build_model(9)
    x = torch.randn(4, 1024, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")