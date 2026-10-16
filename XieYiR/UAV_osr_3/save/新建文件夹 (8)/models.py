import os
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==========================================
# 1. Dataset: 能量谱计算 + 局部标准化 (破除绝对能量捷径)
# ==========================================
class STFTSplitDataset(Dataset):
    def __init__(self, file_paths, labels, target_freq_bins=512):
        self.file_paths = file_paths
        self.labels = labels
        self.target_freq_bins = target_freq_bins  # 抽取滤波后保留的核心频点数 (例如 512)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        label = self.labels[idx]

        with h5py.File(h5_path, 'r') as f:
            real = f['real'][:].T  # [2048, 2055]
            imag = f['imag'][:].T  # [2048, 2055]

            # 1. 瞬时能量计算 (I^2 + Q^2)
            power = real**2 + imag**2

            # ========================================================
            # 物理操作 1: 频域重心对齐 (严格等价于 文献公式 3 载频下变频至 0 频)
            # ========================================================
            # 计算沿时间累加的平均频谱分布
            freq_profile = np.mean(power, axis=1)  # [2048]
            # 找到当前信号的实际中心频率位置 (频点索引)
            center_freq_bin = np.argmax(freq_profile)
            
            # 计算需要滚动的距离，把中心载频强制平移对齐到频域中轴 (1024)
            shift_amount = 1024 - center_freq_bin
            # 在频域维度进行循环搬移 -> 所有已知和未知信号全部在 1024 频点对齐！
            power_shifted = np.roll(power, shift=shift_amount, axis=0)

            # ========================================================
            # 物理操作 2: 7阶滤波与1/4抽取 (频域等价操作: 截取有效带宽)
            # ========================================================
            # 无人机带宽占采样率不超过 1/4 (2048 / 4 = 512 点)，其余均为带外噪声
            # 截取中心 1024 附近的 512 个核心频点，直接斩断带外白噪声！
            half_bw = self.target_freq_bins // 2
            power_filtered = power_shifted[1024 - half_bw : 1024 + half_bw, :]  # [512, 2055]

            # ========================================================
            # 物理操作 3: 信号突发检出与对数标准化
            # ========================================================
            energy = np.log1p(power_filtered)
            
            # 局部标准化 (提升微弱调制纹理对比度)
            mean_v = np.mean(energy)
            std_v = np.std(energy) + 1e-6
            norm_energy = ((energy - mean_v) / std_v).astype(np.float32)

            # 扩充为单通道 -> [1, 512, 2055]
            tensor_data = np.expand_dims(norm_energy, axis=0)

        return torch.from_numpy(tensor_data), label, os.path.basename(h5_path)

# ==========================================
# 2. CNN Block Encoder: 保留 4 个频带子带结构 (避免压平调制指纹)
# ==========================================
class BlockCNNEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 下采样频域
            nn.Conv2d(1, 32, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Stage 2
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(4, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Stage 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # 【升级关键点】不压成 (1, 1)，而是保留 (4, 1) 的频带多通道拓扑
            nn.AdaptiveAvgPool2d((4, 1))
        )
        # 输入维度为 128 * 4 * 1 = 512
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        feat = self.features(x)         # [B * N, 128, 4, 1]
        feat = torch.flatten(feat, 1)   # [B * N, 512]
        return self.fc(feat)            # [B * N, 512]

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
# 5. 单中心 ArcFace 头
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=512, out_features=4, s=20.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feat, label=None, current_margin=None):
        m = self.m if current_margin is None else current_margin
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        cosine = F.linear(F.normalize(input_feat, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        
        if label is None:
            return cosine

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7, max=1.0))
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > th, phi, cosine - mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        margin_logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        margin_logits *= self.s
        return margin_logits, cosine