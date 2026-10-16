"""
CNN + Mamba (Selective State Space Model)
核心：输入依赖的状态空间模型，线性复杂度，长序列建模能力强

纯 PyTorch 实现（无需安装 mamba-ssm 包）
注意：selective scan 使用 Python 循环，速度比官方 CUDA 实现慢
      但 CNN 已将序列压缩到 128，因此循环开销可接受
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_frontend import CNNFrontend, ClassifierHead


class SelectiveSSM(nn.Module):
    """
    Mamba 的核心：选择性状态空间模型

    与传统 SSM (S4) 的区别：
    - S4:  A, B, C 是固定参数 → 对所有输入施加相同的变换
    - Mamba: B, C, Δ 是输入依赖的 → "选择性"地处理不同输入

    状态方程 (离散化后):
        h_t = Ā * h_{t-1} + B̄ * x_t     (状态更新)
        y_t = C_t * h_t                   (输出)
    其中 Ā = exp(Δ_t * A),  B̄ = Δ_t * B_t
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand  # 内部维度扩展

        # 输入投影：分成两路 (x_ssm 走 SSM, z 做门控)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 局部卷积：捕获短程依赖（类似 LSTM 的输入门看近邻）
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner  # Depthwise conv
        )

        # SSM 参数
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A 矩阵（对角化，用 HiPPO 初始化）
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)  # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))  # 存 log 保证正定

        # D 参数（跳跃连接）
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # B, C 投影（输入依赖 → "选择性"的关键）
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, x, dt, B, C):
        """
        选择性扫描（顺序版本）
        x:  (batch, seq_len, d_inner)  输入
        dt: (batch, seq_len, d_inner)  步长（控制"记住多少"）
        B:  (batch, seq_len, d_state)  输入矩阵（输入依赖）
        C:  (batch, seq_len, d_state)  输出矩阵（输入依赖）
        """
        batch, seq_len, d_inner = x.shape

        # 确保在 float32 下计算（AMP 兼容）
        dt = F.softplus(dt.float())
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state), 负数保证稳定

        # 初始化隐状态
        h = torch.zeros(batch, d_inner, self.d_state,
                        device=x.device, dtype=torch.float32)
        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :]   # (B, d_inner)
            x_t = x[:, t, :].float()    # (B, d_inner)
            B_t = B[:, t, :].float()    # (B, d_state)
            C_t = C[:, t, :].float()    # (B, d_state)

            # 离散化
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)           # (B, d_inner, d_state)

            # 状态更新
            h = dA * h + dB * x_t.unsqueeze(-1)

            # 输出
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, seq_len, d_inner)
        # 跳跃连接
        y = y + x.float() * self.D.unsqueeze(0).unsqueeze(0)
        return y.to(x.dtype)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x
        x = self.layer_norm(x)

        # 输入投影 → 两路
        xz = self.in_proj(x)                       # (B, L, d_inner*2)
        x_ssm, z = xz.chunk(2, dim=-1)             # 各 (B, L, d_inner)

        # 局部卷积
        x_ssm = x_ssm.permute(0, 2, 1)             # (B, d_inner, L)
        x_ssm = self.conv1d(x_ssm)[:, :, :x.size(1)]  # 因果裁剪
        x_ssm = x_ssm.permute(0, 2, 1)             # (B, L, d_inner)
        x_ssm = F.silu(x_ssm)

        # 生成输入依赖的 SSM 参数
        dt = self.dt_proj(x_ssm)                    # (B, L, d_inner)
        B = self.B_proj(x_ssm)                      # (B, L, d_state)
        C = self.C_proj(x_ssm)                      # (B, L, d_state)

        # 选择性扫描
        y = self.selective_scan(x_ssm, dt, B, C)

        # 门控输出
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual


class MambaBlock(nn.Module):
    """Mamba 块 = SelectiveSSM + Feed-Forward"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand, dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.ssm(x)
        x = x + self.ff(x)  # 残差 FFN
        return x


class CNN_Mamba(nn.Module):
    """
    架构: CNN(1024→128) → Mamba(4层SSM) → GAP → Classifier
    输入: (batch, 1024, 2)
    """

    def __init__(self, num_classes=9, d_model=128, d_state=16,
                 num_layers=4, dropout=0.2):
        super().__init__()

        self.cnn = CNNFrontend(in_channels=2, out_channels=d_model)

        self.mamba_layers = nn.Sequential(*[
            MambaBlock(d_model, d_state=d_state, d_conv=4, expand=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = ClassifierHead(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 1024, 2)
        x = x.permute(0, 2, 1)          # → (batch, 2, 1024)
        x = self.cnn(x)                  # → (batch, 128, 128)
        x = x.permute(0, 2, 1)          # → (batch, 128, 128) [seq=128, feat=128]
        x = self.mamba_layers(x)         # → (batch, 128, 128)
        x = self.final_norm(x)
        x = x.mean(dim=1)              # GAP → (batch, 128)
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