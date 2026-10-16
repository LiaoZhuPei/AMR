"""
model_transformer_v2.py
同样的双分支思路，把 TCN 换成 Transformer
"""
import math
import torch
import torch.nn as nn
from cnn_frontend_v2 import CNNFrontend
from stat_features import HigherOrderStats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class DualBranchTransformer(nn.Module):
    def __init__(self, num_classes=14, d_model=128, nhead=8,
                 num_layers=4, dim_ff=256, dropout=0.3, stat_dim=64):
        super().__init__()

        # ── 分支1: 时序特征 ──
        self.cnn = CNNFrontend(in_channels=2, out_channels=d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.t_norm = nn.LayerNorm(d_model)

        # ── 分支2: 统计特征 ──
        self.stats = HigherOrderStats(proj_dim=stat_dim)

        # ── 融合分类器 ──
        fused_dim = d_model + stat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        stat_feat = self.stats(x)

        h = x.permute(0, 2, 1)
        h = self.cnn(h)               # (B, d, L//8)
        h = h.permute(0, 2, 1)        # (B, L//8, d)
        h = self.pos_enc(h)
        h = self.transformer(h)
        h = self.t_norm(h)
        h = h.mean(dim=1)             # (B, d)

        out = torch.cat([h, stat_feat], dim=1)
        return self.classifier(out)


def build_model(classes=14):
    return DualBranchTransformer(num_classes=classes)


if __name__ == '__main__':
    model = build_model(14)
    x = torch.randn(4, 2048, 2)
    print(f"Output: {model(x).shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,}")