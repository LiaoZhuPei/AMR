"""
model.py
模型定义 - 单目标 Deep SVDD（独立超球体）

每个目标类别对应一个独立的 SingleSphereModel:
  - 骨干网络提取特征
  - 一个球心 center (1, D)
  - 一个阈值 threshold (标量)
  训练时: 目标类样本向球心聚拢, 负类样本远离球心
  推理时: 计算到球心距离, 与阈值比较判断是否属于该类

EnsemblePredictor:
  汇总 K 个独立模型的距离输出, 做最终决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  骨干网络（结构共享, 权重独立）
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
        prev = 2
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
        return self.proj(self.pool(self.conv(x)).flatten(1))


class CNN2DBackbone(nn.Module):
    """输入 (B, 1, F, T) 时频图"""

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
        return self.proj(self.pool(self.conv(x)).flatten(1))


# ─────────────────────────────────────────────
#  单目标超球体模型
# ─────────────────────────────────────────────

class SingleSphereModel(nn.Module):
    """
    单目标 Deep SVDD 模型。

    训练时:
      - 目标类 (label=1): 特征向球心聚拢
      - 负类   (label=0): 特征远离球心
    推理时:
      - 计算到球心距离, 距离 < threshold → 属于该类

    Attributes:
        target_name: str, 该球体对应的类别名 (如 "A")
    """

    def __init__(self, cfg, target_name=""):
        super().__init__()
        m = cfg["model"]
        self.feature_dim = m["feature_dim"]
        self.backbone_type = m["backbone"]
        self.normalize_feat = m.get("normalize_feature", True)
        self.target_name = target_name

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

        # 单球心 (1, D)
        self.register_buffer("center", torch.zeros(self.feature_dim))
        # 单阈值
        self.register_buffer("threshold", torch.tensor(float("inf")))

    def forward(self, x):
        """提取特征, 返回 (B, D)"""
        f = self.backbone(x)
        if self.normalize_feat:
            f = F.normalize(f, p=2, dim=1, eps=1e-8)
        return f

    def compute_distance(self, feat):
        """到球心的欧氏距离, 返回 (B,)"""
        return torch.sqrt(((feat - self.center) ** 2).sum(dim=1) + 1e-8)

    def compute_distance_sq(self, feat):
        """到球心的距离平方, 返回 (B,)"""
        return ((feat - self.center) ** 2).sum(dim=1)

    @torch.no_grad()
    def init_center(self, data_loader, device, eps=0.1):
        """
        用目标类样本 (label=1) 的特征均值初始化球心。
        data_loader 产出 (x, binary_label, ...), binary_label: 1=目标, 0=负
        """
        self.eval()
        feats = []
        for x, y, *_ in data_loader:
            x = x.to(device)
            f = self.forward(x)
            mask = (y == 1)
            if mask.any():
                feats.append(f[mask].cpu())

        if not feats:
            raise RuntimeError(f"[{self.target_name}] 无目标类样本用于球心初始化")

        center = torch.cat(feats, 0).mean(0)
        center[(center.abs() < eps) & (center < 0)] = -eps
        center[(center.abs() < eps) & (center >= 0)] = eps
        if self.normalize_feat:
            center = F.normalize(center, dim=0, eps=1e-8)
        self.center = center.to(device)
        self.train()
        print(f"    [{self.target_name}] 球心初始化完成, norm={self.center.norm():.4f}")


# ─────────────────────────────────────────────
#  单目标 SVDD 损失
# ─────────────────────────────────────────────

class SingleSphereSVDDLoss(nn.Module):
    """
    L = mean(dist_sq[target])
      + lambda_neg * mean(ReLU(margin - dist_sq[negative]))

    binary_labels: 1=目标类, 0=负类
    """

    def __init__(self, lambda_neg=1.0, margin=4.0):
        super().__init__()
        self.lambda_neg = lambda_neg
        self.margin = margin

    def forward(self, dist_sq, binary_labels):
        target = binary_labels == 1
        negative = binary_labels == 0
        device = dist_sq.device

        loss_pos = dist_sq[target].mean() if target.any() \
            else torch.tensor(0.0, device=device)
        loss_neg = F.relu(self.margin - dist_sq[negative]).mean() if negative.any() \
            else torch.tensor(0.0, device=device)

        return loss_pos + self.lambda_neg * loss_neg


# ─────────────────────────────────────────────
#  集成推理器
# ─────────────────────────────────────────────

class EnsemblePredictor:
    """
    汇总 K 个独立 SingleSphereModel 的输出做最终决策。

    决策规则:
      1. 对每个球体计算距离
      2. 按距离升序排名
      3. 最近球体的距离 < 该球体阈值 → 判定为该类
      4. 所有球体距离均 >= 阈值 → 判定为 unknown (-1)
    """

    def __init__(self, models: dict, device):
        """
        Args:
            models: {类别名: SingleSphereModel} 字典
            device: torch.device
        """
        self.models = models  # {"A": model_A, "D": model_D, ...}
        self.device = device
        self.class_names = list(models.keys())
        self.K = len(self.class_names)

    @torch.no_grad()
    def predict(self, x):
        """
        对一批样本做集成推理。

        Args:
            x: (B, 2, L) 输入

        Returns:
            dict:
              "distances"  : (B, K) 到每个球心的距离
              "ranking"    : (B, K) 距离排名, 0=最近
              "pred_class" : list[str], 长度B, 预测类别名或 "unknown"
              "pred_label" : (B,) 预测标签, -1=unknown
              "is_known"   : (B,) bool
              "details"    : list[dict], 每个样本的详细排名信息
        """
        x = x.to(self.device)
        B = x.size(0)

        # 收集每个球体的距离
        all_dist = torch.zeros(B, self.K, device=self.device)
        all_threshold = torch.zeros(self.K, device=self.device)

        for k, name in enumerate(self.class_names):
            model = self.models[name].eval()
            feat = model(x)
            dist = model.compute_distance(feat)
            all_dist[:, k] = dist
            all_threshold[k] = model.threshold

        # 排名 (距离升序)
        ranking = all_dist.argsort(dim=1)  # (B, K)

        # 决策
        min_dist, min_idx = all_dist.min(dim=1)  # (B,), (B,)
        class_th = all_threshold[min_idx]
        is_known = min_dist < class_th

        pred_label = torch.where(is_known, min_idx, torch.tensor(-1, device=self.device))

        pred_class = []
        details = []
        for b in range(B):
            if is_known[b]:
                pred_class.append(self.class_names[min_idx[b].item()])
            else:
                pred_class.append("unknown")

            # 详细排名信息
            rank_info = []
            for rank_pos in range(self.K):
                k = ranking[b, rank_pos].item()
                name = self.class_names[k]
                d = all_dist[b, k].item()
                th = all_threshold[k].item()
                within = d < th
                rank_info.append({
                    "rank": rank_pos + 1,
                    "class": name,
                    "distance": round(d, 4),
                    "threshold": round(th, 4),
                    "within_sphere": within,
                })
            # 追加 unknown 的"概率"概念: 不属于任何球的程度
            # 用 "最近距离 / 最近阈值" 的比值, >1 表示超出阈值
            ratio = min_dist[b].item() / (class_th[b].item() + 1e-8)
            rank_info.append({
                "unknown_ratio": round(ratio, 4),
                "is_unknown": not is_known[b].item(),
            })
            details.append(rank_info)

        return {
            "distances": all_dist.cpu().numpy(),
            "ranking": ranking.cpu().numpy(),
            "pred_class": pred_class,
            "pred_label": pred_label.cpu().numpy(),
            "is_known": is_known.cpu().numpy(),
            "details": details,
        }