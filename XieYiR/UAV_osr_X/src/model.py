"""
model.py
模型定义 - 多球体 Deep SVDD

骨干:
  - CNN1DBackbone: 输入 (B, 2, L) 处理 IQ 双通道原始信号
  - CNN2DBackbone: 输入 (B, 1, F, T) 处理时频图 (保留)

主模型 MultiSphereDeepSVDDModel:
  骨干网络 → L2归一化特征 → 计算到 K 个球心的距离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
class CNN1DBackbone(nn.Module):
    """输入 (B, 2, L)  I/Q 双通道"""

    def __init__(self, feature_dim=128, channels=None,
                 kernel_sizes=None, pool_sizes=None):
        super().__init__()
        channels = channels or [32, 64, 128]
        kernel_sizes = kernel_sizes or [7, 5, 3]
        pool_sizes = pool_sizes or [2, 2, 2]

        layers = []
        prev = 2  # I/Q 双通道
        for ch, ks, ps in zip(channels, kernel_sizes, pool_sizes):
            layers += [
                nn.Conv1d(prev, ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(ps),
            ]
            prev = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1] * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class CNN2DBackbone(nn.Module):
    """输入 (B, 1, F, T)  时频图"""

    def __init__(self, in_channels=1, feature_dim=128, channels=None):
        super().__init__()
        channels = channels or [16, 32, 64, 128]
        layers, prev = [], in_channels
        for ch in channels:
            layers += [
                nn.Conv2d(prev, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ]
            prev = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(
            nn.Linear(channels[-1] * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ─────────────────────────────────────────────
class MultiSphereDeepSVDDModel(nn.Module):
    """多球体 Deep SVDD 主模型"""

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.feature_dim = m["feature_dim"]
        self.backbone_type = m["backbone"]
        self.num_classes = m["num_classes"]
        self.normalize_feat = m.get("normalize_feature", True)

        if self.backbone_type == "cnn1d":
            c = m["cnn1d"]
            self.backbone = CNN1DBackbone(
                feature_dim=self.feature_dim,
                channels=c["channels"],
                kernel_sizes=c["kernel_sizes"],
                pool_sizes=c["pool_sizes"],
            )
        elif self.backbone_type == "cnn2d":
            c = m["cnn2d"]
            self.backbone = CNN2DBackbone(
                in_channels=1,
                feature_dim=self.feature_dim,
                channels=c["channels"],
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_type}")

        self.register_buffer("centers",
                             torch.zeros(self.num_classes, self.feature_dim))
        self.register_buffer("thresholds",
                             torch.full((self.num_classes,), float("inf")))

    def forward(self, x):
        f = self.backbone(x)
        if self.normalize_feat:
            f = F.normalize(f, p=2, dim=1, eps=1e-8)
        return f

    def compute_distances(self, feat):
        """返回 (B, K) 到每个球心的距离平方"""
        diff = feat.unsqueeze(1) - self.centers.unsqueeze(0)
        return (diff ** 2).sum(dim=2)

    @torch.no_grad()
    def init_centers(self, data_loader, device, eps=0.1):
        """用各类别特征均值初始化球心"""
        self.eval()
        feats = {c: [] for c in range(self.num_classes)}
        for x, labels, *_ in data_loader:
            x = x.to(device)
            f = self.forward(x)
            for c in range(self.num_classes):
                m = labels == c
                if m.any():
                    feats[c].append(f[m].cpu())

        centers = torch.zeros(self.num_classes, self.feature_dim)
        for c in range(self.num_classes):
            if not feats[c]:
                raise RuntimeError(f"Class {c} 无样本用于球心初始化")
            mean = torch.cat(feats[c], 0).mean(0)
            mean[(mean.abs() < eps) & (mean < 0)] = -eps
            mean[(mean.abs() < eps) & (mean >= 0)] = eps
            if self.normalize_feat:
                mean = F.normalize(mean, dim=0, eps=1e-8)
            centers[c] = mean
        self.centers = centers.to(device)
        self.train()


# ─────────────────────────────────────────────
class MultiSphereSVDDLoss(nn.Module):
    """L = mean(own_dist) + λ * mean(ReLU(margin - other_dist))"""

    def __init__(self, lambda_neg=1.0):
        super().__init__()
        self.lambda_neg = lambda_neg

    def forward(self, dist_sq_all, labels, margin=4.0):
        B, K = dist_sq_all.shape
        device = dist_sq_all.device
        idx = torch.arange(B, device=device)
        known = labels >= 0

        if known.any():
            own = dist_sq_all[idx[known], labels[known]]
            loss_pos = own.mean()
        else:
            loss_pos = torch.tensor(0.0, device=device)

        mask_other = torch.ones_like(dist_sq_all, dtype=torch.bool)
        if known.any():
            mask_other[idx[known], labels[known]] = False

        other = dist_sq_all[mask_other]
        loss_neg = F.relu(margin - other).mean() if other.numel() > 0 \
            else torch.tensor(0.0, device=device)

        return loss_pos + self.lambda_neg * loss_neg


class CenterSeparationLoss(nn.Module):
    def __init__(self, margin_center=4.0):
        super().__init__()
        self.margin = margin_center

    def forward(self, centers):
        K = centers.size(0)
        if K < 2:
            return torch.tensor(0.0, device=centers.device)
        d = torch.cdist(centers, centers, p=2) ** 2
        eye = torch.eye(K, dtype=torch.bool, device=centers.device)
        return F.relu(self.margin - d[~eye]).mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat, labels):
        B = feat.size(0)
        if B < 2:
            return torch.tensor(0.0, device=feat.device)
        d = torch.cdist(feat, feat, p=2)
        pos = labels.unsqueeze(1) == labels.unsqueeze(0)
        eye = torch.eye(B, dtype=torch.bool, device=feat.device)
        pos = pos & ~eye
        neg = ~pos & ~eye
        loss = torch.tensor(0.0, device=feat.device)
        if pos.any():
            loss = loss + d[pos].mean()
        if neg.any():
            loss = loss + F.relu(self.margin - d[neg]).mean()
        return loss