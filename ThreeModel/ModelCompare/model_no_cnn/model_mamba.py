"""
纯 Mamba (Selective State Space Model)：直接处理 (batch, 1024, 2)
Linear(2→d_model) → Mamba Blocks ×4 → GAP → Classifier

Mamba 本身就是为长序列设计的，1024 对它来说很短
唯一的问题：纯 Python 的 selective scan 用 for 循环，1024 步比 128 步慢 ~8 倍
           如果安装了 mamba-ssm 官方 CUDA 包，则几乎没有速度损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Mamba 核心：选择性状态空间模型"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner
        )

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def selective_scan(self, x, dt, B, C):
        """选择性扫描（顺序实现）"""
        batch, seq_len, d_inner = x.shape

        dt = F.softplus(dt.float())
        A = -torch.exp(self.A_log.float())

        h = torch.zeros(batch, d_inner, self.d_state,
                        device=x.device, dtype=torch.float32)
        outputs = []

        for t in range(seq_len):
            dt_t = dt[:, t, :]
            x_t = x[:, t, :].float()
            B_t = B[:, t, :].float()
            C_t = C[:, t, :].float()

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h = dA * h + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x.float() * self.D.unsqueeze(0).unsqueeze(0)
        return y.to(x.dtype)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = self.conv1d(x_ssm)[:, :, :residual.size(1)]
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = F.silu(x_ssm)

        dt = self.dt_proj(x_ssm)
        B = self.B_proj(x_ssm)
        C = self.C_proj(x_ssm)

        y = self.selective_scan(x_ssm, dt, B, C)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y + residual


class MambaBlock(nn.Module):
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
        x = x + self.ff(x)
        return x


class PureMamba(nn.Module):
    """
    输入: (batch, 1024, 2)
    无 CNN，直接用 Linear 投影 2 → d_model，然后 Mamba 处理全部 1024 步
    """

    def __init__(self, num_classes=9, d_model=128, d_state=16,
                 num_layers=4, dropout=0.2):
        super().__init__()

        # 输入投影：2 → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.mamba_layers = nn.Sequential(*[
            MambaBlock(d_model, d_state=d_state, d_conv=4, expand=2, dropout=dropout)
            for _ in range(num_layers)
        ])

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
        x = self.input_proj(x)          # → (batch, 1024, d_model)
        x = self.mamba_layers(x)        # → (batch, 1024, d_model)
        x = self.final_norm(x)
        x = x.mean(dim=1)             # GAP → (batch, d_model)
        return self.classifier(x)


def build_model(classes=9):
    return PureMamba(num_classes=classes)


if __name__ == '__main__':
    model = build_model(9)
    x = torch.randn(4, 1024, 2)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")