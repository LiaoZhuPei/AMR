import os
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# =========================================================================
# 1. Dataset: 纯净原始 STFT 读取 [2, 2048, 2055] (无需任何人工破坏性增强)
# =========================================================================
class STFTSplitDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        label = self.labels[idx]

        with h5py.File(h5_path, 'r') as f:
            real = f['real'][:].T
            imag = f['imag'][:].T
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# =========================================================================
# 2. 基础残差块
# =========================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

# =========================================================================
# 3. 经典全局渐进下采样骨干网络 (GlobalSpectrogramResNet)
# =========================================================================
class GlobalSpectrogramResNet(nn.Module):
    def __init__(self, in_channels=2, feat_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = nn.Sequential(ConvBlock(32, 64, stride=2), ConvBlock(64, 64, stride=1))
        self.stage2 = nn.Sequential(ConvBlock(64, 128, stride=2), ConvBlock(128, 128, stride=1))
        self.stage3 = nn.Sequential(ConvBlock(128, 256, stride=2), ConvBlock(256, 256, stride=1))
        self.stage4 = nn.Sequential(ConvBlock(256, 512, stride=2), ConvBlock(512, 512, stride=1))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        feats = self.head(out)
        # 强制归一化到单位超球表面
        return F.normalize(feats, p=2, dim=1)

# =========================================================================
# 4. ArcFace 角度超锥投影层与损失函数 (Additive Angular Margin)
# =========================================================================
class ArcMarginProduct(nn.Module):
    """
    ArcFace 核心实现：
    在超球面角度空间上强制压缩类内方差，拉开类间间隔。
    让 E 的特征聚集成极窄的锐角光束，彻底切断 C 的渗入空间。
    """
    def __init__(self, in_features=512, out_features=4, s=32.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # 尺度缩放因子
        self.m = m  # 角度间隔 margin (0.50 rad 约等于 28.6 度)

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 1. 归一化权重与特征，计算余弦相似度 cos(theta)
        cosine = F.linear(input, F.normalize(self.weight, p=2, dim=1))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # 2. 如果是测试/推理阶段（无标签），直接返回真实余弦相似度
        if label is None:
            return cosine

        # 3. 如果是训练阶段，施加加性角度惩罚: cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 构建 one-hot 掩码，只对目标类别增加角度惩罚
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output