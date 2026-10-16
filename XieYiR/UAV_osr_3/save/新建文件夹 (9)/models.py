import os
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==========================================
# 1. Dataset: 原始纯净 STFT 读取 (彻底取消一切外部经验预处理)
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
            # 1. 读出并转置为 [2048, 2055]
            real = f['real'][:].T  # shape: (2048, 2055)
            imag = f['imag'][:].T  # shape: (2048, 2055)
            
            # 2. 堆叠为双通道 (2, 2048, 2055)
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# ==========================================
# 2. CNN Block Encoder: [2, 2048, 32] -> 512
# ==========================================
class BlockCNNEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            # 输入通道恢复为 2 (实部与虚部)
            nn.Conv2d(2, 32, kernel_size=(7, 3), stride=(4, 1), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(4, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.fc(feat)

# ==========================================
# 3. Temporal Transformer Encoder
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TemporalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return torch.mean(x, dim=1)  # 全局序列嵌入 [Batch, 512]

# ==========================================
# 4. 全局特征提取器
# ==========================================
class FullFeatureExtractor(nn.Module):
    def __init__(self, win_size=32, stride=16, d_model=512):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.block_cnn = BlockCNNEncoder(out_dim=d_model)
        self.temporal_transformer = TemporalTransformer(d_model=d_model)

    def forward(self, x):
        B, C, F_dim, T = x.shape
        blocks = x.unfold(dimension=3, size=self.win_size, step=self.stride)
        N_blocks = blocks.shape[3]
        blocks = blocks.permute(0, 3, 1, 2, 4).contiguous().view(B * N_blocks, C, F_dim, self.win_size)
        
        z = self.block_cnn(blocks)
        z = z.view(B, N_blocks, -1)
        seq_embed = self.temporal_transformer(z)
        return seq_embed

# ==========================================
# 5. 修复尺度后的 ARPLayer (维度归一化，彻底根治 Loss 爆炸)
# ==========================================
class ARPLayer(nn.Module):
    def __init__(self, num_classes=4, feat_dim=512, temp=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temp = temp  # 黄金温度系数

        # 互易点权重初始化
        self.points = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.normal_(self.points, mean=0.0, std=1.0)

        # 初始半径设为 1.5 (匹配维度均方距离尺度)
        self.radius = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))

    def forward(self, z):
        # z: [Batch, 512]
        # 计算维度平均欧氏距离平方 (除以 feat_dim=512 防止高维距离爆炸)
        z_sq = torch.sum(z ** 2, dim=1, keepdim=True)             # [B, 1]
        p_sq = torch.sum(self.points ** 2, dim=1).unsqueeze(0)      # [1, C]
        dist = (z_sq - 2.0 * torch.matmul(z, self.points.t()) + p_sq) / self.feat_dim
        dist = torch.clamp(dist, min=1e-7)  # 保证数值在 0.5 ~ 3.0 的健康区间

        # 互易分类 Logits
        logits = dist / self.temp
        return logits, dist

# ==========================================
# 6. 修复尺度后的 ARPLoss
# ==========================================
class ARPLoss(nn.Module):
    def __init__(self, weight_osb=0.5, weight_adv=0.1, eps=0.05):
        super().__init__()
        self.weight_osb = weight_osb
        self.weight_adv = weight_adv
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, dist, labels, arp_layer, z):
        # 1. 互易分类损失 (约 1.38 左右)
        loss_cls = self.ce(logits, labels)

        # 2. 开放空间有界约束 (OSB): 控制在 0 ~ 1 之间
        target_dist = dist.gather(1, labels.view(-1, 1)).squeeze(1)
        loss_osb = torch.mean(F.relu(target_dist - arp_layer.radius) ** 2)

        # 3. 对抗互易扰动
        if self.weight_adv > 0 and arp_layer.training:
            z_var = z.detach().requires_grad_(True)
            l_var, _ = arp_layer(z_var)
            loss_temp = self.ce(l_var, labels)
            z_grad = torch.autograd.grad(loss_temp, z_var)[0]

            # 生成对抗样本
            z_adv = z_var + self.eps * torch.sign(z_grad)

            # 对抗样本距离同样除以 feat_dim
            z_adv_sq = torch.sum(z_adv ** 2, dim=1, keepdim=True)
            p_sq = torch.sum(arp_layer.points ** 2, dim=1).unsqueeze(0)
            dist_adv = (z_adv_sq - 2.0 * torch.matmul(z_adv, arp_layer.points.t()) + p_sq) / arp_layer.feat_dim
            dist_adv = torch.clamp(dist_adv, min=1e-7)
            loss_adv = torch.mean(dist_adv)
        else:
            loss_adv = torch.tensor(0.0, device=dist.device)

        total_loss = loss_cls + self.weight_osb * loss_osb + self.weight_adv * loss_adv
        return total_loss