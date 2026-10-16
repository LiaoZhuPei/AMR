"""
CNN + Temporal Convolutional Network (TCN)
核心：膨胀卷积，全并行，通过指数增长的 dilation 覆盖长距离依赖
仅注入 A 通道 + 双池化 (Ablation Study Version)
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
    def __init__(self, num_classes=9, channels=128, num_layers=7,
                 kernel_size=3, dropout=0.2):
        super().__init__()

        # 前端输入变为 3 通道 (I, Q, A)
        self.cnn = CNNFrontend(in_channels=3, out_channels=channels)

        tcn_blocks = []
        for i in range(num_layers):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32
            tcn_blocks.append(TCNBlock(channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_blocks)

        self.final_norm = nn.BatchNorm1d(channels)

        # 【修改点】：去掉了物理特征后，分类头输入维度仅为双池化的拼接结果
        # Mean Pooling (128) + Max Pooling (128) = 256
        self.classifier = ClassifierHead(channels * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x 初始形状: (batch, 1024, 2)
        x_perm = x.permute(0, 2, 1)  # -> (batch, 2, 1024)

        I = x_perm[:, 0:1, :]  # 保持维度 (batch, 1, 1024)
        Q = x_perm[:, 1:2, :]

        # 1. 显式计算幅度 A
        # (注意：这里使用 1e-4 防止 autocast 混合精度下梯度爆炸引发警告)
        A = torch.sqrt(I ** 2 + Q ** 2 + 1e-4)

        # 2. 把 A 作为第 3 个通道和 I, Q 拼起来喂给 CNN
        cnn_input = torch.cat([I, Q, A], dim=1)  # -> (batch, 3, 1024)

        # 3. 深度网络前向传播
        feat = self.cnn(cnn_input)  # 注意这里喂的是 3 通道的 cnn_input
        feat = self.tcn(feat)
        feat = self.final_norm(feat)

        # 4. 双池化 (Dual Pooling)
        # 均值池化：看整体趋势； 最大池化：敏锐捕捉 256QAM 的罕见极值点
        feat_mean = feat.mean(dim=2)  # (batch, 128)
        feat_max = feat.max(dim=2)[0]  # (batch, 128)

        # 融合池化特征
        feat_pooled = torch.cat([feat_mean, feat_max], dim=1)  # -> (batch, 256)

        # 5. 直接输出分类结果
        return self.classifier(feat_pooled)


def build_model(classes=9):
    return CNN_TCN(num_classes=classes)


if __name__ == '__main__':
    model = build_model(9)
    x = torch.randn(4, 1024, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")