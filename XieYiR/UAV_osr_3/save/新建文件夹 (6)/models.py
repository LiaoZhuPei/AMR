import os
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==========================================
# 1. Dataset: 计算能量谱，以单通道 [1, 2048, 2055] 输出
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
            real = f['real'][:].T  # [2048, 2055]
            imag = f['imag'][:].T  # [2048, 2055]

            # 1. 计算时频能量谱 (Power / Energy): I^2 + Q^2
            # 2. 采用 log1p 进行动态范围平滑压缩，防止强载波淹没弱时序特征，同时增强数值稳定性
            energy = np.log1p(real**2 + imag**2).astype(np.float32)
            
            # 3. 扩充通道维度 -> [1, 2048, 2055]
            tensor_data = np.expand_dims(energy, axis=0)

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# ==========================================
# 2. CNN Block Encoder: 单通道输入 [1, 2048, 32] -> 512
# ==========================================
class BlockCNNEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            # 第一层输入通道修改为 1
            nn.Conv2d(1, 32, kernel_size=(7, 3), stride=(4, 1), padding=(3, 1)),
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
# 4. 全局特征提取器 (接收单通道能量时频图)
# ==========================================
class FullFeatureExtractor(nn.Module):
    def __init__(self, win_size=32, stride=16, d_model=512):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.block_cnn = BlockCNNEncoder(out_dim=d_model)
        self.temporal_transformer = TemporalTransformer(d_model=d_model)

    def forward(self, x):
        B, C, F_dim, T = x.shape  # C = 1
        blocks = x.unfold(dimension=3, size=self.win_size, step=self.stride)
        N_blocks = blocks.shape[3]
        blocks = blocks.permute(0, 3, 1, 2, 4).contiguous().view(B * N_blocks, C, F_dim, self.win_size)
        
        z = self.block_cnn(blocks)
        z = z.view(B, N_blocks, -1)
        seq_embed = self.temporal_transformer(z)
        return seq_embed

# ==========================================
# 5. 单中心 ArcFace 度量学习头 (回归单中心，边界纯净)
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=512, out_features=4, s=20.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # 单中心权重: [out_features, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feat, label=None, current_margin=None):
        m = self.m if current_margin is None else current_margin
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        # 归一化特征与权重 -> 计算超球面余弦相似度 [-1.0, 1.0]
        cosine = F.linear(F.normalize(input_feat, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        
        # 测试与推断模式：直接返回余弦矩阵
        if label is None:
            return cosine

        # 训练阶段：添加角度边际惩罚
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7, max=1.0))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        margin_logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        margin_logits *= self.s

        return margin_logits, cosine