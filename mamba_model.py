"""
mamba_model.py
===================
基于 Mamba (Selective State Space Model) 的调制识别模型
输入格式: (batch, 1024, 2)  — 与原 CNN_BiLSTM_Attention 相同
输出格式: (batch, num_classes)

依赖安装（需要 CUDA）：
    pip install mamba-ssm causal-conv1d

若 GPU 环境无法安装 mamba-ssm，可使用本文件内置的
纯 PyTorch Mamba 实现（MambaPure），无需额外依赖。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 纯 PyTorch 版 Mamba Block（无需 mamba-ssm 包）
# 参考论文: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
# ============================================================

class MambaBlock(nn.Module):
    """
    单个 Mamba Block（S6 选择性扫描）的纯 PyTorch 实现。

    参数:
        d_model   : 输入/输出特征维度
        d_state   : SSM 状态维度 (N)，默认 16
        d_conv    : 因果卷积核大小，默认 4
        expand    : 扩展因子，内部维度 = d_model * expand，默认 2
        dt_rank   : Δ 的秩，默认 ceil(d_model / 16)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = int(d_model * expand)     # D_inner
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # 输入投影：一次线性投影产生 x 和 z
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 因果 1D 卷积（depthwise）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # x_proj: 从 x 投影出 Δ, B, C
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + d_state * 2,  # dt_rank + N + N
            bias=False
        )

        # dt_proj: 将低秩 Δ 升维到 d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: log(-A) 参数，shape (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: 跳跃连接系数
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 归一化
        self.norm = nn.LayerNorm(d_model)

    def ssm(self, x):
        """
        选择性状态空间扫描（顺序实现，适用于训练推理）
        x: (batch, L, d_inner)
        返回: (batch, L, d_inner)
        """
        B_sz, L, d_inner = x.shape
        d_state = self.d_state

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 获取 Δ, B_ssm, C_ssm
        x_dbl = self.x_proj(x)             # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, d_state, d_state], dim=-1)

        # Δ softplus 激活并升维
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        B_ssm = B_ssm                      # (B, L, d_state)
        C_ssm = C_ssm                      # (B, L, d_state)

        # 离散化（ZOH）: Ā = exp(Δ·A), B̄ = Δ·B
        # 逐步扫描实现（parallel scan 可进一步加速，这里用循环简洁实现）
        # dt:   (B, L, d_inner)
        # A:    (d_inner, d_state)
        # B_ssm:(B, L, d_state)

        # 转为 (B, d_inner, L) 便于矩阵运算
        dt_T   = dt.permute(0, 2, 1)          # (B, d_inner, L)
        B_ssm_T = B_ssm.permute(0, 2, 1)      # (B, d_state, L)
        C_ssm_T = C_ssm.permute(0, 2, 1)      # (B, d_state, L)
        x_T    = x.permute(0, 2, 1)           # (B, d_inner, L)

        # Ā = exp(Δ·A)  → (B, d_inner, d_state, L)
        # 使用 einsum: dA[b,i,n,l] = dt_T[b,i,l] * A[i,n]
        dA = torch.einsum('bil,in->binl', dt_T, A)   # (B, d_inner, d_state, L)
        dA = torch.exp(dA)

        # B̄ = Δ·B  → (B, d_inner, d_state, L)
        dB = torch.einsum('bil,bnl->binl', dt_T, B_ssm_T)  # (B, d_inner, d_state, L)

        # 顺序扫描：h_t = Ā·h_{t-1} + B̄·u_t
        h = torch.zeros(B_sz, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for l in range(L):
            h = dA[:, :, :, l] * h + dB[:, :, :, l] * x_T[:, :, l].unsqueeze(-1)
            # y_t = C·h_t  → (B, d_inner)
            y_l = torch.einsum('bin,bn->bi', h, C_ssm_T[:, :, l])
            ys.append(y_l)

        y = torch.stack(ys, dim=2)  # (B, d_inner, L)
        y = y.permute(0, 2, 1)     # (B, L, d_inner)

        # 加入 D 跳跃连接
        y = y + x * self.D

        return y

    def forward(self, x):
        """
        x: (batch, L, d_model)
        返回: (batch, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # 线性投影 → x_branch, z_gate
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_branch, z_gate = xz.chunk(2, dim=-1)        # 各 (B, L, d_inner)

        # 因果卷积（需要去掉 causal padding 的右侧多余部分）
        x_conv = x_branch.permute(0, 2, 1)            # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :x_conv.shape[-1]]  # causal
        x_conv = x_conv.permute(0, 2, 1)              # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM
        y = self.ssm(x_conv)                          # (B, L, d_inner)

        # 门控
        y = y * F.silu(z_gate)                        # (B, L, d_inner)

        # 输出投影
        output = self.out_proj(y)                     # (B, L, d_model)

        return output + residual


# ============================================================
# 可选：使用官方 mamba-ssm 包（更快的 CUDA kernel）
# ============================================================
def _try_import_mamba():
    """尝试导入官方 mamba-ssm；失败则返回 None。"""
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        return None


# ============================================================
# Mamba 调制识别模型
# ============================================================

class MambaModulationClassifier(nn.Module):
    """
    Mamba 调制识别模型

    架构：
        输入 (batch, 1024, 2)
        → 输入嵌入层（Linear + 残差）将 2 维 IQ 映射到 d_model
        → N 个 Mamba Block（序列建模）
        → 全局平均池化 + 最大池化（双路聚合）
        → 分类头（MLP）
        → 输出 (batch, num_classes)

    参数:
        num_classes : 分类数，默认 9
        d_model     : Mamba 隐藏维度，默认 128
        n_layers    : Mamba Block 数量，默认 4
        d_state     : SSM 状态维度，默认 16
        d_conv      : 因果卷积核大小，默认 4
        expand      : 内部扩展比，默认 2
        dropout     : 分类头 dropout，默认 0.3
        use_official: 若为 True 且 mamba-ssm 已安装，则使用官方 CUDA kernel
    """

    def __init__(
        self,
        num_classes=9,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.3,
        use_official=True
    ):
        super().__init__()

        # ---------- 输入嵌入 ----------
        # IQ 两通道 → d_model，用 1D 卷积提取局部特征做初始嵌入
        self.input_embed = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size=7, padding=3),   # 局部感知
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        # 输出: (batch, d_model, 1024)

        # ---------- Mamba 块 ----------
        # 决定使用官方实现还是纯 PyTorch 实现
        OfficialMamba = _try_import_mamba() if use_official else None

        blocks = []
        for _ in range(n_layers):
            if OfficialMamba is not None:
                block = OfficialMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            else:
                block = MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            blocks.append(block)

        self.mamba_layers = nn.ModuleList(blocks)

        # 每层之间加 LayerNorm（官方 Mamba 内部已有 norm，纯实现也含）
        # 这里再加一层最终 norm
        self.final_norm = nn.LayerNorm(d_model)

        # ---------- 分类头 ----------
        # 双路聚合：平均池化 + 最大池化，拼接后 MLP
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 4),
            nn.Linear(128, num_classes)
        )

        # ---------- 权重初始化 ----------
        self._init_weights()

        # 打印使用哪种 Mamba 实现
        mode = "官方 mamba-ssm CUDA kernel" if OfficialMamba is not None else "纯 PyTorch 实现"
        print(f"[MambaModulationClassifier] 使用 {mode}，共 {n_layers} 层，d_model={d_model}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (batch, 1024, 2)  — 与原模型输入格式完全一致
        """
        # (batch, 1024, 2) → (batch, 2, 1024)  适配 Conv1d
        x = x.permute(0, 2, 1)

        # 输入嵌入
        x = self.input_embed(x)        # (batch, d_model, 1024)

        # (batch, d_model, 1024) → (batch, 1024, d_model)  适配 Mamba
        x = x.permute(0, 2, 1)

        # 逐层 Mamba Block
        for layer in self.mamba_layers:
            x = layer(x)               # (batch, 1024, d_model)

        x = self.final_norm(x)         # (batch, 1024, d_model)

        # 双路全局聚合
        avg_pool = x.mean(dim=1)       # (batch, d_model)
        max_pool = x.max(dim=1).values # (batch, d_model)
        x = torch.cat([avg_pool, max_pool], dim=-1)  # (batch, d_model*2)

        # 分类
        output = self.classifier(x)    # (batch, num_classes)
        return output


def build_model(classes=9, d_model=128, n_layers=4, d_state=16,
                d_conv=4, expand=2, dropout=0.3, use_official=True):
    """
    构建 Mamba 调制识别模型。

    参数:
        classes     : 分类数
        d_model     : Mamba 隐藏维度（建议 64~256）
        n_layers    : Mamba Block 数量（建议 2~6）
        d_state     : SSM 状态维度（建议 8~32）
        d_conv      : 因果卷积核大小
        expand      : 内部扩展比
        dropout     : 分类头 dropout
        use_official: 优先使用官方 CUDA kernel（需安装 mamba-ssm）
    """
    return MambaModulationClassifier(
        num_classes=classes,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        use_official=use_official
    )


# ============================================================
# 快速测试
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = build_model(classes=9).to(device)

    # 测试输入格式：(batch, 1024, 2)  — 与训练代码一致
    x = torch.randn(4, 1024, 2).to(device)
    out = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
