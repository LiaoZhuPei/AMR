"""
model_transformer_v2.py v2
双分支: CNN→Transformer (时序) + HigherOrderStats (统计) → 融合
stat_dim=96, 配合 stat_features.py v3 (42维输入)
"""
import math
import torch
import torch.nn as nn
from cnn_frontend_v2 import CNNFrontend
from stat_features import HigherOrderStats


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])


class DualBranchTransformer(nn.Module):
    """
    分支1: CNN(8x下采样) → PositionalEncoding → TransformerEncoder → GAP → 128维
    分支2: HigherOrderStats(42维原始 → 96维投影)
    融合:  Cat(128+96=224) → 384 → 192 → num_classes
    """

    def __init__(self, num_classes=14, d_model=128, nhead=8,
                 num_layers=4, dim_ff=256, dropout=0.3, stat_dim=96):
        super().__init__()

        # ── 分支1: 时序特征 ──
        self.cnn = CNNFrontend(in_channels=2, out_channels=d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-Norm 更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.t_norm = nn.LayerNorm(d_model)

        # ── 分支2: 统计特征 (42维 → 96维) ──
        self.stats = HigherOrderStats(proj_dim=stat_dim)

        # ── 融合分类器 ──
        fused_dim = d_model + stat_dim  # 128 + 96 = 224
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        # x: (B, L, 2)

        # 分支2: 统计特征（从原始 IQ 提取）
        stat_feat = self.stats(x)              # (B, 96)

        # 分支1: 时序特征
        h = x.permute(0, 2, 1)                # (B, 2, L)
        h = self.cnn(h)                        # (B, 128, L//8)
        h = h.permute(0, 2, 1)                # (B, L//8, 128)
        h = self.pos_enc(h)                    # + 位置编码
        h = self.transformer(h)                # (B, L//8, 128)
        h = self.t_norm(h)
        h = h.mean(dim=1)                     # GAP → (B, 128)

        # 融合
        out = torch.cat([h, stat_feat], dim=1) # (B, 224)
        return self.classifier(out)


def build_model(classes=14):
    return DualBranchTransformer(num_classes=classes)


if __name__ == '__main__':
    model = build_model(14)
    print(model)

    # 测试 2048 长度
    x = torch.randn(4, 2048, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    # 测试 1024 长度
    x2 = torch.randn(4, 1024, 2)
    out2 = model(x2)
    print(f"Input: {x2.shape} → Output: {out2.shape}")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")