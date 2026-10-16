"""
CNN + Transformer Encoder
核心：Multi-Head Self-Attention，全并行，全局关系建模
"""
import math
import torch
import torch.nn as nn
from cnn_frontend import CNNFrontend, ClassifierHead


class PositionalEncoding(nn.Module):
    """正弦位置编码：给序列中每个位置赋予唯一的位置信息"""

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNN_Transformer(nn.Module):
    """
    架构: CNN(1024→128) → Positional Encoding → Transformer Encoder → GAP → Classifier
    输入: (batch, 1024, 2)
    """

    def __init__(self, num_classes=9, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=256, dropout=0.3):
        super().__init__()

        self.cnn = CNNFrontend(in_channels=2, out_channels=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=256, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-Norm（训练更稳定）
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = ClassifierHead(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 1024, 2)
        x = x.permute(0, 2, 1)               # → (batch, 2, 1024)
        x = self.cnn(x)                       # → (batch, 128, 128)
        x = x.permute(0, 2, 1)               # → (batch, 128, 128)  [seq_len=128, d=128]
        x = self.pos_encoder(x)               # + 位置编码
        x = self.transformer_encoder(x)       # → (batch, 128, 128)
        x = self.final_norm(x)
        x = x.mean(dim=1)                    # Global Average Pooling → (batch, 128)
        return self.classifier(x)


def build_model(classes=9):
    return CNN_Transformer(num_classes=classes)


if __name__ == '__main__':
    model = build_model(9)
    x = torch.randn(4, 1024, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")