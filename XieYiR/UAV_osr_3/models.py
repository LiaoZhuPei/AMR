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

        with h5py.File(h5_path, "r") as f:
            # 读出并转置为 [2048, 2055]
            real = f["real"][:].T
            imag = f["imag"][:].T
            # 堆叠为 (2, 2048, 2055)
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)


# ==========================================
# 2. HDC 混合膨胀因果/非因果时序卷积块 (HDC-TCN Block)
# ==========================================
class HDCBlock1D(nn.Module):
    """单个 HDC 残差块：

    采用指定的 dilation rate，padding=dilation 严格保持时间轴尺寸不变。
    """

    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        # 卷积核 k=3，padding=dilation 确保输入与输出在时间轴上点对点对齐
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        return self.act2(out + residual)


# ==========================================
# 3. 方案 B：时频 HDC-TCN 全局时序骨干网络 (SpectrogramHDCTCN)
# ==========================================
class SpectrogramHDCTCN(nn.Module):

    def __init__(self, in_channels=2, out_dim=512, hidden_dim=512):
        super().__init__()
        # -----------------------------------------------------------
        # Step 1. 浅层 2D Stem: 快速抑制超大尺度的显存开销
        # [2, 2048, 2055] -> [32, 256, 257]
        # -----------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=7, stride=4, padding=3, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # -----------------------------------------------------------
        # Step 2. 频域折叠编码器 (Frequency Encoder):
        # 仅在频域维度(Freq)进行下采样融合，绝对不损伤时间轴维度(Time=257)!
        # [32, 256, 257] -> [64, 64, 257] -> [128, 16, 257] -> [512, 1, 257]
        # -----------------------------------------------------------
        self.freq_encoder = nn.Sequential(
            # Freq: 256 -> 64
            nn.Conv2d(
                32, 64, kernel_size=(5, 1), stride=(4, 1), padding=(2, 0)
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Freq: 64 -> 16
            nn.Conv2d(
                64, 128, kernel_size=(5, 1), stride=(4, 1), padding=(2, 0)
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Freq: 16 -> 1，将整个频带的高阶特征映射至通道中
            nn.Conv2d(
                128,
                hidden_dim,
                kernel_size=(16, 1),
                stride=(1, 1),
                padding=(0, 0),
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # -----------------------------------------------------------
        # Step 3. HDC-TCN 核心层 (Hybrid Dilated Convolution):
        # 采用 3 组循环的锯齿互质膨胀序列: [1, 2, 3, 5]
        # 相邻膨胀层公约数为 1，彻底消灭感受野网格空洞 (Gridding Artifacts)
        # -----------------------------------------------------------
        hdc_dilations = [1, 2, 3, 5] * 3  # 循环3次，共12个残差块
        hdc_layers = []
        for d in hdc_dilations:
            hdc_layers.append(HDCBlock1D(channels=hidden_dim, dilation=d))
        self.tcn_backbone = nn.Sequential(*hdc_layers)

        # -----------------------------------------------------------
        # Step 4. 时序全局池化与表征头
        # -----------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 汇聚 257 个时间步
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: [Batch, 2, 2048, 2055]
        # 1. 2D 浅层提取 -> [Batch, 32, 256, 257]
        feat2d = self.stem(x)

        # 2. 频域压缩 -> [Batch, 512, 1, 257]
        feat_time = self.freq_encoder(feat2d)

        # 3. 维度重整为纯时序 1D 格式 -> [Batch, 512, 257]
        feat_time = feat_time.squeeze(2)

        # 4. HDC-TCN 时序长程依赖建模 (时序长度始终保持 257，感受野覆盖全图)
        out_tcn = self.tcn_backbone(feat_time)  # [Batch, 512, 257]

        # 5. 全局时序汇聚与分类嵌入映射
        out_pooled = self.global_pool(out_tcn).squeeze(-1)  # [Batch, 512]
        out_embed = self.head(out_pooled)  # [Batch, 512]

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

        # 初始半径设为 1.5 (匹配均方距离尺度)
        self.radius = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def forward(self, z):
        # 维度均方距离: ||z - P||^2 / feat_dim，彻底阻断损失爆炸
        z_sq = torch.sum(z**2, dim=1, keepdim=True)
        p_sq = torch.sum(self.points**2, dim=1).unsqueeze(0)
        dist = (
            z_sq - 2.0 * torch.matmul(z, self.points.t()) + p_sq
        ) / self.feat_dim
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

            z_adv_sq = torch.sum(z_adv**2, dim=1, keepdim=True)
            p_sq = torch.sum(arp_layer.points**2, dim=1).unsqueeze(0)
            dist_adv = (
                z_adv_sq
                - 2.0 * torch.matmul(z_adv, arp_layer.points.t())
                + p_sq
            ) / arp_layer.feat_dim
            dist_adv = torch.clamp(dist_adv, min=1e-7)
            loss_adv = torch.mean(dist_adv)
        else:
            loss_adv = torch.tensor(0.0, device=dist.device)

        total_loss = (
            loss_cls + self.weight_osb * loss_osb + self.weight_adv * loss_adv
        )
        return total_loss