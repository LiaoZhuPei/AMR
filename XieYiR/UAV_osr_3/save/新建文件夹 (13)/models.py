import os
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# =========================================================================
# 1. Dataset: 注入频域镜像增强的纯净 STFT 读取
# =========================================================================
class STFTSplitDataset(Dataset):
    def __init__(self, file_paths, labels, is_train=False):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        label = self.labels[idx]

        with h5py.File(h5_path, 'r') as f:
            real = f['real'][:].T # [2048, 2055]
            imag = f['imag'][:].T
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        # 核心突破：训练阶段 50% 概率沿频率轴 (axis=1) 垂直镜像翻转
        # 强制打破 E 的高频垄断，逼迫 CNN 提取纯粹的微观时序脉冲指纹
        if self.is_train and np.random.rand() < 0.5:
            tensor_data = np.ascontiguousarray(np.flip(tensor_data, axis=1))

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# =========================================================================
# 2. 原版稳健残差块
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
# 3. 原版全局 ResNet 骨干 (最快、最稳、不掉点)
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
        return F.normalize(feats, p=2, dim=1) # 严格归一化到单位超球面

# =========================================================================
# 4. ArcFace 角度超锥投影层
# =========================================================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=512, out_features=4, s=32.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(input, F.normalize(self.weight, p=2, dim=1))
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        if label is None:
            return cosine

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output