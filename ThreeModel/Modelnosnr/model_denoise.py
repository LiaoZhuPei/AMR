# -*- coding: utf-8 -*-
"""
model_denoise.py
一维全卷积残差去噪网络 (IQ-DenoiseNet)
输入形状: (Batch, 2, L) -> 输出形状: (Batch, 2, L)
"""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """一维时域残差块"""

    def __init__(self, channels):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)  # 引入残差学习，保持相位一致性


class IQDenoiseNet(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, num_blocks=5):
        super(IQDenoiseNet, self).__init__()

        # 时域特征提取层 (Batch, 2, L) -> (Batch, 64, L)
        self.in_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

        # 串联 5 个深度残差单元，用以恢复相位旋转与去除畸变
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock1D(hidden_channels))
        self.residual_layers = nn.Sequential(*blocks)

        # 时域特征还原层 (Batch, 64, L) -> (Batch, 2, L)
        self.out_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=7, padding=3)

    def forward(self, x):
        out = self.relu(self.in_conv(x))
        out = self.residual_layers(out)
        out = self.out_conv(out)
        return out