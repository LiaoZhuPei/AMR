import os
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==========================================
# 1. Dataset: 文件级独立读取
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
# 2. CNN Block Encoder: [2, 2048, 32] -> 512
# ==========================================
class BlockCNNEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
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
        return torch.mean(x, dim=1)

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
# 5. Metric Learning: Sub-Center ArcFace 头
# ==========================================
class SubCenterArcMarginProduct(nn.Module):
    def __init__(self, in_features=512, out_features=4, K=3, s=20.0, m=0.35):
        """
        in_features: 512 维特征
        out_features: 已知类别数 (如 4)
        K: 每个类别分配的子中心数量 (如 3)
        s: 超球缩放系数
        m: 边际角度惩罚
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.s = s
        self.m = m

        # 权重维度: [out_features * K, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feat, label=None, current_margin=None):
        m = self.m if current_margin is None else current_margin
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        # 1. 归一化输入特征与全部 C * K 个子中心
        x_norm = F.normalize(input_feat, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # 2. 计算与所有子中心的余弦相似度: [Batch, out_features * K]
        cosine_all = F.linear(x_norm, w_norm)
        B = x_norm.size(0)

        # 变型为: [Batch, out_features, K]
        cosine_all = cosine_all.view(B, self.out_features, self.K)

        # 3. 聚合机制：每个类别选取与其最接近的那个子中心 (Max Pooling over K)
        # raw_cosine: [Batch, out_features]，每一项即代表该样本到该类最近子中心的相似度
        raw_cosine, _ = torch.max(cosine_all, dim=2)

        # 推断模式：直接返回每个类别的最大相似度，用于判定已知或未知
        if label is None:
            return raw_cosine

        # 4. 训练阶段: 只对真实目标类别施加边际惩罚
        target_cosine = raw_cosine.gather(1, label.view(-1, 1))  # [B, 1]
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(target_cosine, 2), min=1e-7, max=1.0))
        phi = target_cosine * cos_m - sine * sin_m
        phi = torch.where(target_cosine > th, phi, target_cosine - mm)

        one_hot = torch.zeros_like(raw_cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        margin_logits = (one_hot * phi) + ((1.0 - one_hot) * raw_cosine)
        margin_logits *= self.s

        return margin_logits, raw_cosine