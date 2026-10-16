import os
import glob
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ==========================================
# 1. Dataset: 读取 h5 并根据首字母分配标签
# ==========================================
class STFTH5Dataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: train 文件夹路径
        文件命名规则: 首字母即类别, 如 'A_01.h5', 'a_02.h5' 均属于类别 'A'
        """
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if len(self.file_paths) == 0:
            raise ValueError(f"No .h5 files found in {data_dir}")

        # 提取首字母作为类别并映射为连续的数字 ID
        raw_labels = [os.path.basename(p)[0].upper() for p in self.file_paths]
        self.classes = sorted(list(set(raw_labels)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.labels = [self.class_to_idx[l] for l in raw_labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        label = self.labels[idx]

        with h5py.File(h5_path, 'r') as f:
            # 1. 读出并转置为 [2048, 2055]
            real = f['real'][:].T  # shape: (2048, 2055)
            imag = f['imag'][:].T  # shape: (2048, 2055)
            
            # 2. 堆叠为 (2, 2048, 2055)
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        # 3. 转为 PyTorch Tensor [2, 2048, 2055]
        return torch.from_numpy(tensor_data), label

# ==========================================
# 2. CNN Block Encoder: 单个时频块特征提取 [2, 2048, 32] -> 512
# ==========================================
class BlockCNNEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        # 频点 2048 较大，通过步长卷积和池化快速下采样频域
        self.features = nn.Sequential(
            # Stage 1: Conv -> BatchNorm -> Activation
            nn.Conv2d(2, 32, kernel_size=(7, 3), stride=(4, 1), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Stage 2: Conv -> Pooling
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(4, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Stage 3: Conv -> Global Pooling
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Stage 4: Linear -> z_i [out_dim]
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: [Batch * N, 2, 2048, 32]
        feat = self.features(x)         # [Batch * N, 128, 1, 1]
        feat = torch.flatten(feat, 1)   # [Batch * N, 128]
        z = self.fc(feat)               # [Batch * N, 512]
        return z

# ==========================================
# 3. Temporal Transformer Encoder
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TemporalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, N, 512]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        # 序列聚合: Mean Pooling 获得整段序列的 Sequence Embedding
        seq_embedding = torch.mean(x, dim=1)  # [Batch, 512]
        return seq_embedding

# ==========================================
# 4. 端到端组合模型 (Sliding Window + CNN + Transformer)
# ==========================================
class FullFeatureExtractor(nn.Module):
    def __init__(self, win_size=32, stride=16, d_model=512):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.block_cnn = BlockCNNEncoder(out_dim=d_model)
        self.temporal_transformer = TemporalTransformer(d_model=d_model)

    def forward(self, x):
        # x: [Batch, 2, 2048, 2055]
        B, C, F, T = x.shape

        # 1. 采用 unfold 在时间维度滑窗切分: [Batch, 2, 2048, N_blocks, win_size]
        # N_blocks = (2055 - 32) // 16 + 1 = 127
        blocks = x.unfold(dimension=3, size=self.win_size, step=self.stride)
        N_blocks = blocks.shape[3]

        # 调整为: [Batch, N_blocks, 2, 2048, win_size] -> 重构为 [Batch * N, 2, 2048, win_size]
        blocks = blocks.permute(0, 3, 1, 2, 4).contiguous()
        blocks = blocks.view(B * N_blocks, C, F, self.win_size)

        # 2. Block-level CNN 提取特征
        z = self.block_cnn(blocks)  # [Batch * N, 512]
        z = z.view(B, N_blocks, -1) # [Batch, N, 512]

        # 3. Transformer 学习时间帧依赖
        seq_embed = self.temporal_transformer(z) # [Batch, 512]
        return seq_embed