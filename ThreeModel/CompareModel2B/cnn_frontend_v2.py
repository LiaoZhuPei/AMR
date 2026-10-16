"""
cnn_frontend_v2.py
多尺度 CNN + SE 通道注意力 + 残差
输入: (B, 2, L)   输出: (B, C, L//8)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, ch, r=4):
        super().__init__()
        mid = max(ch // r, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False), nn.ReLU(True),
            nn.Linear(mid, ch, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x).flatten(1)).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """1D 残差块"""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CNNFrontend(nn.Module):
    """
    多尺度 CNN 前端
    - 三种 kernel (3/7/15) 捕捉不同时间尺度
    - SE 注意力让模型学习"哪些特征通道重要"
    - 残差连接稳定训练
    - 3 次 stride=2 → 总共 8x 下采样
    """
    def __init__(self, in_channels=2, out_channels=128):
        super().__init__()
        # 多尺度初始卷积
        self.b3  = nn.Conv1d(in_channels, 32, 3,  padding=1,  bias=False)
        self.b7  = nn.Conv1d(in_channels, 32, 7,  padding=3,  bias=False)
        self.b15 = nn.Conv1d(in_channels, 32, 15, padding=7,  bias=False)
        self.bn0 = nn.BatchNorm1d(96)

        # Stage 1: 96 → 64, L → L/2
        self.s1 = nn.Sequential(
            nn.Conv1d(96, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            ResBlock(64), SEBlock(64),
        )
        # Stage 2: 64 → C, L/2 → L/4
        self.s2 = nn.Sequential(
            nn.Conv1d(64, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.GELU(),
            ResBlock(out_channels), SEBlock(out_channels),
        )
        # Stage 3: C → C, L/4 → L/8
        self.s3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.GELU(),
            ResBlock(out_channels), SEBlock(out_channels),
        )

    def forward(self, x):
        x = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        x = F.gelu(self.bn0(x))
        return self.s3(self.s2(self.s1(x)))


class ClassifierHead(nn.Module):
    """分类头 (供不使用统计特征的模型调用)"""
    def __init__(self, in_features, num_classes, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)