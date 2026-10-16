import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_mlp(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()

        self.linear_e1 = nn.Linear(128, 256)
        self.batchnorm1d = nn.BatchNorm1d(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear_e2 = nn.Linear(256,128)



    def forward(self, x):
        x = self.linear_e1(x)
        x = x.permute(0, 2, 1)
        x = self.batchnorm1d(x)
        x = x.permute(0, 2, 1)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_e2(x)
        return x

class PositionalEncoding(nn.Module):

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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        x_tanh = torch.tanh(x)
        scores = self.w(x_tanh)
        alpha = F.softmax(scores, dim=1)

        weighted_x = alpha * x

        h_pool = torch.sum(weighted_x, dim=1)

        return h_pool

class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=8, dropout=0.3):
        super().__init__()
        self.attention_pooling = AttentionPooling(hidden_dim=hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 256)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, 128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, num_classes)
        # self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.attention_pooling(x)
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.batchnorm2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        # x = self.softmax(x)

        return x

class Decoder_mlp(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.batchnorm1d = nn.BatchNorm1d(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256 * 8, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = x.permute(0, 2, 1)
        x = self.batchnorm1d(x)
        x = x.permute(0, 2, 1)
        x = self.gelu(x)
        x = self.dropout(x)

        x = x.flatten(1, 2)

        x = self.linear2(x)
        x = x.reshape(-1, 2, 256)

        return x

class MyModule(nn.Module):
    def __init__(self, dropout=0.1, d_model=128, nhead=8, dim_feedforward=256, num_layers=2):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.encoder_mlp = Encoder_mlp(dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len=256, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        self.decoder_mlp = Decoder_mlp(dropout=dropout)
        self.classification = ClassificationHead(dropout=dropout)

    def forward(self, x, class_mode):
        # x:(batch, 2, 256)
        x = x.view(-1, 2, 8, 32)
        x = x.permute(0, 2, 1, 3)
        x = x.flatten(2, 3)
        x = self.linear1(x)
        x = self.encoder_mlp(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        if not class_mode:
            x = self.decoder_mlp(x)
        else:
            x = self.classification(x)


        return x

def build_model():
    return MyModule

if __name__ == '__main__':
    model = build_model()

    print(f"I-I-I")