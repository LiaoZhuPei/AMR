import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==========================================
# 1. Dataset: 纯净原始 STFT 读取 [2, 2048, 2055]
# ==========================================
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

# ==========================================
# 2. 残差基础块
# ==========================================
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

# ==========================================
# 3. 骨干残差网络 (输出单位归一化特征，杜绝特征发散)
# ==========================================
class GlobalSpectrogramResNet(nn.Module):
    def __init__(self, in_channels=2, feat_dim=256):
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
        # L2 归一化：将特征约束在超球面上，使得欧氏距离具有极其严格的物理尺度
        feats = F.normalize(feats, p=2, dim=1)
        return feats

# ==========================================
# 4. 紧凑原型分类器 (Prototype Classifier)
# ==========================================
class PrototypeLayer(nn.Module):
    def __init__(self, num_classes=4, feat_dim=256, scale=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.scale = scale # 缩放因子，控制 Softmax 锐度
        
        # 每一个已知类一个原型中心
        self.prototypes = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.orthogonal_(self.prototypes) # 正交初始化，让各已知类初始距离最大化

    def forward(self, z):
        # 保证原型也在单位球面上
        proto = F.normalize(self.prototypes, p=2, dim=1)
        
        # 计算归一化特征 z 与原型之间的欧氏距离平方
        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2(z·c) = 2 - 2(z·c)
        sim = torch.matmul(z, proto.t()) # 余弦相似度 [-1, 1]
        dist = 2.0 - 2.0 * sim # 欧氏距离平方 [0, 4]
        dist = torch.clamp(dist, min=0.0)

        # 转换为分类对数几率 (距离越近，logit 越高)
        logits = -self.scale * dist
        return logits, dist

# ==========================================
# 5. 紧凑原型损失 (Compact Prototype Loss)
# ==========================================
class PrototypeLoss(nn.Module):
    def __init__(self, weight_compact=0.5):
        super().__init__()
        self.weight_compact = weight_compact
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, dist, labels):
        loss_ce = self.ce(logits, labels)
        
        # 紧凑损失：强制已知类紧紧聚拢在自己球心周围，压缩开放空间
        target_dist = dist.gather(1, labels.view(-1, 1)).squeeze(1)
        loss_compact = torch.mean(target_dist)

        total_loss = loss_ce + self.weight_compact * loss_compact
        return total_loss