import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==========================================
# 1. Dataset: 原始 STFT 读取 [2, 2048, 2055]
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
            # 读出并转置为 [2048, 2055]
            real = f['real'][:].T
            imag = f['imag'][:].T
            # 堆叠为 (2, 2048, 2055)
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# ==========================================
# 2. 残差基础块 (BasicBlock for Spectrogram)
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
        self.shortcut = nn.Sequential() # 形状一致（stride == 1 且 in_c == out_c）,保持空的 nn.Sequential()，输入x直通输出
        if stride != 1 or in_c != out_c:
            # 形状不一致
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

# ==========================================
# 3. 方案 A：全局渐进下采样 2D-CNN (GlobalSpectrogramResNet)
# ==========================================
class GlobalSpectrogramResNet(nn.Module):
    def __init__(self, in_channels=2, out_dim=512):
        super().__init__()
        # Stem: 快速下采样大尺度时频矩阵，降低显存
        # [2, 2048, 2055] -> [32, 512, 514] -> [32, 256, 257]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 级残差层次拓扑，感受野逐级放大至全图
        # Stage 1: [32, 256, 257] -> [64, 128, 129]
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 64, stride=1)
        )
        # Stage 2: [64, 128, 129] -> [128, 64, 65]
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128, stride=1)
        )
        # Stage 3: [128, 64, 65] -> [256, 32, 33]
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256, stride=1)
        )
        # Stage 4: [256, 32, 33] -> [512, 16, 17]
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, stride=2),
            ConvBlock(512, 512, stride=1)
        )

        # 全局池化与映射头
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: [Batch, 2, 2048, 2055] 单次直接输入全景时频图
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        
        out = self.global_pool(out)      # [Batch, 512, 1, 1]
        out = torch.flatten(out, 1)      # [Batch, 512]
        out_embed = self.head(out)        # [Batch, 512]
        return out_embed

# ==========================================
# 4. 对抗互易点分类层 (ARPLayer，带维度均方根治)
# ==========================================
class ARPLayer(nn.Module):
    def __init__(self, num_classes=4, feat_dim=512, temp=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temp = temp

        # 每个已知类对应一个互易反空间中心
        self.points = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.normal_(self.points, mean=0.0, std=1.0)
        # 这里两句，torch.FloatTensor和nn.init.normal_直接等同于torch.randn

        # 初始半径设为 1.5 (匹配均方距离尺度)
        self.radius = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def forward(self, z):
        # 维度均方距离: ||z - P||^2 / feat_dim，彻底阻断损失爆炸
        z_sq = torch.sum(z ** 2, dim=1, keepdim=True)
        p_sq = torch.sum(self.points ** 2, dim=1).unsqueeze(0)
        dist = (z_sq - 2.0 * torch.matmul(z, self.points.t()) + p_sq) / self.feat_dim
        dist = torch.clamp(dist, min=1e-7)

        logits = dist / self.temp
        return logits, dist

# ==========================================
# 5. 对抗互易点损失函数 (ARPLoss)
# ==========================================
class ARPLoss(nn.Module):
    def __init__(self, weight_osb=0.5, weight_adv=0.1, eps=0.05):
        super().__init__()
        self.weight_osb = weight_osb
        self.weight_adv = weight_adv
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, dist, labels, arp_layer, z):
        # 1. 互易分类损失
        loss_cls = self.ce(logits, labels)

        # 2. 开放空间有界收缩 (OSB)
        target_dist = dist.gather(1, labels.view(-1, 1)).squeeze(1)
        loss_osb = torch.mean(F.relu(target_dist - arp_layer.radius) ** 2)

        # 3. 对抗扰动项 (攻防边界收缩)
        if self.weight_adv > 0 and arp_layer.training:
            z_var = z.detach().requires_grad_(True)
            l_var, _ = arp_layer(z_var)
            loss_temp = self.ce(l_var, labels)
            z_grad = torch.autograd.grad(loss_temp, z_var)[0]

            z_adv = z_var + self.eps * torch.sign(z_grad)

            z_adv_sq = torch.sum(z_adv ** 2, dim=1, keepdim=True)
            p_sq = torch.sum(arp_layer.points ** 2, dim=1).unsqueeze(0)
            dist_adv = (z_adv_sq - 2.0 * torch.matmul(z_adv, arp_layer.points.t()) + p_sq) / arp_layer.feat_dim
            dist_adv = torch.clamp(dist_adv, min=1e-7)
            loss_adv = torch.mean(dist_adv)
        else:
            loss_adv = torch.tensor(0.0, device=dist.device)

        total_loss = loss_cls + self.weight_osb * loss_osb + self.weight_adv * loss_adv
        return total_loss