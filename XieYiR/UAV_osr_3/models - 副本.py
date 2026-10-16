import os
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# =========================================================================
# 1. Dataset: 纯净原始 STFT 读取 [2, 2048, 2055]
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

        # 训练时 50% 镜像翻转，打破绝对高频垄断
        if self.is_train and np.random.rand() < 0.5:
            tensor_data = np.ascontiguousarray(np.flip(tensor_data, axis=1))

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# =========================================================================
# 2. 基础残差块
# =========================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

# =========================================================================
# 3. 时-频双流动力学网络 (DTS-ResNet)
#    彻底保留时间脉冲时序，从物理根源上分离 C 和 E
# =========================================================================
class DTS_SpectrogramResNet(nn.Module):
    def __init__(self, in_channels=2, feat_dim=512):
        super().__init__()
        # Stem: [2, 2048, 2055] -> [32, 512, 514] -> [32, 256, 257]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=(4, 4), padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)
        )

        # Stage 1: [32, 256, 257] -> [64, 128, 129]
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, stride=(2, 2)),
            ConvBlock(64, 64, stride=(1, 1))
        )
        # Stage 2: 保持频率 128 精度 [64, 128, 129] -> [128, 128, 65]
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=(1, 2)),
            ConvBlock(128, 128, stride=(1, 1))
        )
        # Stage 3: [128, 128, 65] -> [256, 128, 33]
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=(1, 2)),
            ConvBlock(256, 256, stride=(1, 1))
        )
        # Stage 4: [256, 128, 33] -> [512, 128, 33] (保留丰富时序步长，不再下采样时间)
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, stride=(1, 1)),
            ConvBlock(512, 512, stride=(1, 1))
        )

        # -------------------------------------------------------------
        # 核心创新：时-频双流动力学池化头
        # -------------------------------------------------------------
        # 支路 1: 频域结构流 (提取精确信道频带) -> 输出 256 维
        self.freq_stream = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 支路 2: 时域脉冲流 (专门提取脉冲宽度、跳频间隔与占空比) -> 输出 256 维
        self.time_stream = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 双流融合头 (256 + 256 = 512)
        self.head = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )

    def forward(self, x):
        # x: [B, 2, 2048, 2055]
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out) # [B, 512, F=128, T=33]

        # 1. 频域支路：沿时间轴压缩，提取 128 个频带分布
        f_profile = torch.mean(out, dim=-1) # [B, 512, 128]
        f_feat = self.freq_stream(f_profile).squeeze(-1) # [B, 256]

        # 2. 时域支路：沿频率轴压缩，提取 33 步脉冲跳变包络
        t_profile = torch.mean(out, dim=-2) # [B, 512, 33]
        t_feat = self.time_stream(t_profile).squeeze(-1) # [B, 256]

        # 3. 双流物理融合：兼备频带位置与脉冲微观动态
        fused = torch.cat([f_feat, t_feat], dim=1) # [B, 512]
        
        feats = self.head(fused)
        return F.normalize(feats, p=2, dim=1)

# =========================================================================
# 4. ArcFace 超紧角锥层 (回调至最稳健的 m=0.50, s=32.0)
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