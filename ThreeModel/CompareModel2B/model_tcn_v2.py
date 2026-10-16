"""
model_tcn_v2.py v2
双分支: CNN→TCN (时序) + HigherOrderStats (统计) → 融合
改动: stat_dim 64→96, 给统计分支更多表达力
"""
import torch
import torch.nn as nn
from cnn_frontend_v2 import CNNFrontend
from stat_features import HigherOrderStats


class TCNBlock(nn.Module):
    def __init__(self, ch, ks, dilation, dropout=0.2):
        super().__init__()
        pad = dilation * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, ks, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class DualBranchTCN(nn.Module):
    def __init__(self, num_classes=14, channels=128, tcn_layers=6,
                 kernel_size=3, dropout=0.2, stat_dim=96):  # ← stat_dim: 64→96
        super().__init__()

        # ── 分支1: 时序特征 ──
        self.cnn = CNNFrontend(in_channels=2, out_channels=channels)
        self.tcn = nn.Sequential(*[
            TCNBlock(channels, kernel_size, 2 ** i, dropout)
            for i in range(tcn_layers)
        ])
        self.t_norm = nn.BatchNorm1d(channels)

        # ── 分支2: 统计特征 (32维 → 96维) ──
        self.stats = HigherOrderStats(proj_dim=stat_dim)

        # ── 融合分类器 ──
        fused_dim = channels + stat_dim  # 128 + 96 = 224
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(192, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: (B, L, 2)
        stat_feat = self.stats(x)       # (B, 96)

        h = x.permute(0, 2, 1)         # (B, 2, L)
        h = self.cnn(h)                 # (B, 128, L//8)
        h = self.tcn(h)                 # (B, 128, L//8)
        h = self.t_norm(h)
        h = h.mean(dim=2)              # GAP → (B, 128)

        out = torch.cat([h, stat_feat], dim=1)  # (B, 224)
        return self.classifier(out)


def build_model(classes=14):
    return DualBranchTCN(num_classes=classes)


if __name__ == '__main__':
    model = build_model(14)
    x = torch.randn(4, 2048, 2)
    print(f"Output: {model(x).shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,}")