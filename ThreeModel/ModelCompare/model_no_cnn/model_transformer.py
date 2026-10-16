"""
纯 Transformer：直接处理 (batch, 1024, 2)
Linear(2→d_model) → Positional Encoding → Transformer Encoder → GAP → Classifier

注意：Self-Attention 复杂度 O(L²·d)，L=1024 时每层需要 1024×1024 的注意力矩阵
     可行，但比 TCN/Mamba 更吃显存
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PureTransformer(nn.Module):
    """
    输入: (batch, 1024, 2)
    无 CNN，直接用 Linear 将特征维度 2 → d_model
    """

    def __init__(self, num_classes=9, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=256, dropout=0.3):
        super().__init__()

        # 输入投影：2 → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=2048, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, 1024, 2)
        x = self.input_proj(x)                # → (batch, 1024, d_model)
        x = self.pos_encoder(x)               # + 位置编码
        x = self.transformer_encoder(x)       # → (batch, 1024, d_model)
        x = self.final_norm(x)
        x = x.mean(dim=1)                    # GAP → (batch, d_model)
        return self.classifier(x)


def build_model(classes=9):
    return PureTransformer(num_classes=classes)


if __name__ == '__main__':
    model = build_model(9)
    x = torch.randn(4, 1024, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")