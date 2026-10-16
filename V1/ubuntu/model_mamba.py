import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_frontend import CNNFrontend, ClassifierHead
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Mamba 块 = Pre-Norm + SelectiveSSM + Pre-Norm + FFN"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        # ✅ Pre-Norm 结构：先归一化再送入SSM，数值更稳定
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        # ✅ SSM输出后也加dropout
        self.ssm_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ✅ Pre-Norm + 残差，避免SSM输入过大
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = self.ssm_dropout(x)   # ✅ SSM后dropout
        x = residual + x  # 残差连接

        # FFN分支
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        return x


class CNN_Mamba(nn.Module):
    """
    架构: CNN(1024→128) → Mamba(4层SSM) → GAP → Classifier
    输入: (batch, 1024, 2)
    """

    def __init__(self, num_classes=9, d_model=128, d_state=16,
                 num_layers=4, dropout=0.3):   # ✅ dropout 0.2→0.3
        super().__init__()

        self.cnn = CNNFrontend(in_channels=2, out_channels=d_model)
        # ✅ 进入Mamba前先做归一化
        self.input_norm = nn.LayerNorm(d_model)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model,
                d_state=d_state,
                d_conv=4,
                expand=2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        # ✅ 分类头加强正则：增加dropout
        # ✅ 简化分类头，减少参数
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, num_classes)
        )

        # ✅ 权重初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # ✅ 更小的初始化gain
                nn.init.xavier_uniform_(module.weight, gain=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


    def forward(self, x):
        # x: (batch, 1024, 2)
        x = x.permute(0, 2, 1)          # → (batch, 2, 1024)
        x = self.cnn(x)                  # → (batch, 128, 128)
        x = x.permute(0, 2, 1)          # → (batch, 128, 128) [seq=128, feat=128]

        # ✅ 输入归一化
        x = self.input_norm(x)

        # ✅ 逐层forward（替代nn.Sequential，便于调试）
        for layer in self.mamba_layers:
            x = layer(x)                 # → (batch, 128, 128)

        x = self.final_norm(x)
        x = x.mean(dim=1)               # GAP → (batch, 128)
        return self.classifier(x)


def build_model(classes=9):
    return CNN_Mamba(num_classes=classes)


if __name__ == '__main__':
    model = build_model(9)
    x = torch.randn(4, 1024, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")