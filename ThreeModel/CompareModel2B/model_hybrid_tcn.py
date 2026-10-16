"""
model_hybrid_tcn.py
===================
三分支混合输入模型

Branch A — IQ 原始序列 (1024, 2)
  CNN Frontend (与原 model_tcn.py 相同) → TCN × 6 → GAP → (128,)

Branch B — 瞬时特征序列 (3, 1024)
  轻量 1-D CNN (3层) → GAP → (64,)

Branch C — 循环/统计特征向量 (FEAT_DIM,)
  FC(FEAT_DIM → 128) → BN → GELU → FC(128 → 64) → (64,)

Fusion
  concat(128+64+64) = (256,)
  → FC(256 → 256) → BN → GELU
  → SE 注意力门控 (channel-wise squeeze-excitation)
  → Dropout → Linear(256 → num_classes)

设计原则：
  · Branch B 和 C 均为"纯可选"：若 use_branch_b=False / use_branch_c=False
    则自动跳过，退化为纯 IQ 模型（兼容旧权重加载逻辑）
  · 不依赖 OpenCV、不产生图片文件
  · 特征维度由 hybrid_features.FEAT_DIM 自动读取
"""

import torch
import torch.nn as nn
from hybrid_features import FEAT_DIM, INST_CHANNELS


# ══════════════════════════════════════════════════════════════════════════
# 辅助模块
# ══════════════════════════════════════════════════════════════════════════

class SEGate(nn.Module):
    """Squeeze-Excitation 通道注意力门控"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C)
        return x * self.gate(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# Branch A — CNN Frontend (与原始保持一致，支持加载旧权重)
# ══════════════════════════════════════════════════════════════════════════

class CNNFrontend(nn.Module):
    """
    IQ 序列压缩: (B, 2, 1024) → (B, out_channels, 128)
    共 3 个 stride=2 的卷积块，压缩 8×
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 128):
        super().__init__()
        ch = out_channels
        self.net = nn.Sequential(
            # 1024 → 512
            nn.Conv1d(in_channels, ch // 4, 7, stride=2, padding=3),
            nn.BatchNorm1d(ch // 4), nn.GELU(),
            # 512 → 256
            nn.Conv1d(ch // 4, ch // 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(ch // 2), nn.GELU(),
            # 256 → 128
            nn.Conv1d(ch // 2, ch, 3, stride=2, padding=1),
            nn.BatchNorm1d(ch), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)   # (B, 128, 128)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class BranchA(nn.Module):
    """IQ 序列 → (B, 128)"""

    def __init__(self, channels: int = 128, num_layers: int = 6,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.cnn = CNNFrontend(in_channels=2, out_channels=channels)
        self.tcn = nn.Sequential(*[
            TCNBlock(channels, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x: (B, 1024, 2)
        x = x.permute(0, 2, 1)   # → (B, 2, 1024)
        x = self.cnn(x)           # → (B, 128, 128)
        x = self.tcn(x)           # → (B, 128, 128)
        x = self.norm(x)
        return x.mean(dim=2)      # GAP → (B, 128)


# ══════════════════════════════════════════════════════════════════════════
# Branch B — 瞬时特征 1-D CNN → (B, 64)
# ══════════════════════════════════════════════════════════════════════════

class BranchB(nn.Module):
    """
    输入: (B, INST_CHANNELS=3, 1024)
    输出: (B, 64)
    """

    def __init__(self, in_ch: int = INST_CHANNELS, out_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            # 1024 → 512
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            # 512 → 128
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            # 128 → 32
            nn.Conv1d(64, 64, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(dropout),
            # 32 → 8
            nn.Conv1d(64, out_dim, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(out_dim), nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 3, 1024)
        x = self.net(x)     # (B, out_dim, ~8)
        return x.mean(dim=2)  # GAP → (B, 64)


# ══════════════════════════════════════════════════════════════════════════
# Branch C — 循环/统计特征 MLP → (B, 64)
# ══════════════════════════════════════════════════════════════════════════

class BranchC(nn.Module):
    """
    输入: (B, FEAT_DIM=20)
    输出: (B, 64)
    """

    def __init__(self, in_dim: int = FEAT_DIM, out_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)   # (B, 64)


# ══════════════════════════════════════════════════════════════════════════
# 主模型
# ══════════════════════════════════════════════════════════════════════════

class HybridTCN(nn.Module):
    """
    三分支混合输入 AMR 模型

    forward 参数:
      iq_seq   (B, 1024, 2)   — 原始 IQ (必须)
      inst     (B, 3, 1024)   — 瞬时特征 (Branch B，可选)
      cyclic   (B, FEAT_DIM)  — 循环统计 (Branch C，可选)

    若 use_branch_b=False / use_branch_c=False，对应分支输出 zeros，
    相当于退化为 IQ-only 推理（无需重新训练纯 IQ 模型）。
    """

    def __init__(self,
                 num_classes:  int   = 14,
                 channels:     int   = 128,
                 tcn_layers:   int   = 6,
                 kernel_size:  int   = 3,
                 dropout:      float = 0.2,
                 use_branch_b: bool  = True,
                 use_branch_c: bool  = True):
        super().__init__()

        self.use_b = use_branch_b
        self.use_c = use_branch_c

        # ---- 三个分支 ----
        self.branch_a = BranchA(channels, tcn_layers, kernel_size, dropout)

        if use_branch_b:
            self.branch_b = BranchB(INST_CHANNELS, 64, dropout)

        if use_branch_c:
            self.branch_c = BranchC(FEAT_DIM, 64, dropout)

        # ---- 融合维度计算 ----
        fuse_in = channels                         # A: 128
        if use_branch_b:
            fuse_in += 64                          # B: +64
        if use_branch_c:
            fuse_in += 64                          # C: +64
        # fuse_in ∈ {128, 192, 256}

        fuse_out = 256
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fuse_out),
            nn.BatchNorm1d(fuse_out),
            nn.GELU(),
        )
        self.se     = SEGate(fuse_out, reduction=8)
        self.head   = ClassifierHead(fuse_out, num_classes, dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, iq_seq, inst=None, cyclic=None):
        """
        iq_seq : (B, 1024, 2)
        inst   : (B, 3, 1024)   or None
        cyclic : (B, FEAT_DIM)  or None
        """
        feats = [self.branch_a(iq_seq)]

        if self.use_b:
            if inst is None:
                feats.append(torch.zeros(iq_seq.size(0), 64, device=iq_seq.device))
            else:
                feats.append(self.branch_b(inst))

        if self.use_c:
            if cyclic is None:
                feats.append(torch.zeros(iq_seq.size(0), 64, device=iq_seq.device))
            else:
                feats.append(self.branch_c(cyclic))

        x = torch.cat(feats, dim=1)   # (B, fuse_in)
        x = self.fusion(x)            # (B, 256)
        x = self.se(x)                # (B, 256) with channel attention
        return self.head(x)           # (B, num_classes)


# ══════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════

def build_model(classes: int = 14,
                use_branch_b: bool = True,
                use_branch_c: bool = True) -> HybridTCN:
    return HybridTCN(
        num_classes   = classes,
        channels      = 128,
        tcn_layers    = 6,
        kernel_size   = 3,
        dropout       = 0.2,
        use_branch_b  = use_branch_b,
        use_branch_c  = use_branch_c,
    )


# ══════════════════════════════════════════════════════════════════════════
# 单元测试
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from hybrid_features import FEAT_DIM, INST_CHANNELS
    B = 4

    model = build_model(classes=14)
    iq     = torch.randn(B, 1024, 2)
    inst   = torch.randn(B, INST_CHANNELS, 1024)
    cyclic = torch.randn(B, FEAT_DIM)

    out = model(iq, inst, cyclic)
    print(f'Input IQ:     {iq.shape}')
    print(f'Input inst:   {inst.shape}')
    print(f'Input cyclic: {cyclic.shape}')
    print(f'Output:       {out.shape}')

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total:,}')

    # 测试退化模式（仅 IQ）
    model_iq_only = build_model(classes=14, use_branch_b=False, use_branch_c=False)
    out2 = model_iq_only(iq)
    print(f'\nIQ-only mode output: {out2.shape}')
    total2 = sum(p.numel() for p in model_iq_only.parameters() if p.requires_grad)
    print(f'IQ-only parameters:  {total2:,}')
