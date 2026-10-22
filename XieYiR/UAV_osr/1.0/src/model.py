"""
model.py
模型定义模块（多球体 Deep SVDD 版本）

包含:
  - CNN2DBackbone: 2D-CNN 特征提取器（用于时频图）
  - CNN1DBackbone: 1D-CNN 特征提取器（直接处理原始信号，输入为电压值序列）
  - MultiSphereDeepSVDDModel: 完整模型（骨干网络 + 投影头 + 多球心参数）

多球体 Deep SVDD 核心思想:
  为每个已知类别学习一个独立球心 c_k，训练时让类别 k 的样本聚拢到 c_k 附近，
  同时被其他类别球心排斥。推理时计算样本到所有球心的距离，取最近球心作为
  预测类别；若最近距离仍超过该类别对应阈值，则判定为“未知类”（开集拒识）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  2D-CNN 骨干（用于频谱图输入）
# ─────────────────────────────────────────────

class CNN2DBackbone(nn.Module):
    """
    轻量级 2D-CNN，输入: (B, 1, freq_bins, time_frames)
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128,
                 channels: list = None):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128]

        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers += [
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ]
            prev_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten_dim = channels[-1] * 4 * 4
        self.proj = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


# ─────────────────────────────────────────────
#  1D-CNN 骨干（直接处理原始信号：单通道电压值序列）
# ─────────────────────────────────────────────

class CNN1DBackbone(nn.Module):
    """
    1D-CNN，输入: (B, 1, signal_length)
    输入为原始时域电压采样值（非IQ双通道），因此固定为单输入通道。
    """

    def __init__(self, feature_dim: int = 128,
                 channels: list = None,
                 kernel_sizes: list = None,
                 pool_sizes: list = None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        if pool_sizes is None:
            pool_sizes = [2, 2, 2]

        layers = []
        prev_ch = 1  # 单通道电压序列
        for ch, ks, ps in zip(channels, kernel_sizes, pool_sizes):
            layers += [
                nn.Conv1d(prev_ch, ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(ps)
            ]
            prev_ch = ch

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        self.flatten_dim = channels[-1] * 16
        self.proj = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


# ─────────────────────────────────────────────
#  多球体 Deep SVDD 完整模型
# ─────────────────────────────────────────────

class MultiSphereDeepSVDDModel(nn.Module):
    """
    多球体 Deep SVDD 开放集识别模型。

    组件:
      1. 骨干网络 (CNN2D 或 CNN1D)：提取判别特征
      2. 特征归一化 (在 forward 中做 L2 normalize，约束到单位超球面)
      3. K 个可学习球心 `centers` (K, D)，不参与梯度，warm-up 后统计初始化
      4. 每类独立阈值 `thresholds` (K,)，推理阶段由验证集标定

    训练策略:
      Phase 1 (warm-up): 用全部已知类样本做多分类式预训练，
                          让骨干网络先学到判别性特征
      Phase 2:           用 MultiSphereSVDDLoss 做球心收缩 + 类间排斥，
                          可选加入球心分离约束 (CenterSeparationLoss)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        self.feature_dim = model_cfg["feature_dim"]
        self.backbone_type = model_cfg["backbone"]
        self.num_classes = model_cfg["num_classes"]          # 已知类别数 K
        self.normalize_feat = model_cfg.get("normalize_feature", True)

        # ── 选择骨干 ──
        if self.backbone_type == "cnn2d":
            self.backbone = CNN2DBackbone(
                in_channels=1,
                feature_dim=self.feature_dim,
                channels=model_cfg["cnn2d"]["channels"]
            )
        elif self.backbone_type == "cnn1d":
            self.backbone = CNN1DBackbone(
                feature_dim=self.feature_dim,
                channels=model_cfg["cnn1d"]["channels"],
                kernel_sizes=model_cfg["cnn1d"]["kernel_sizes"],
                pool_sizes=model_cfg["cnn1d"]["pool_sizes"]
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_type}")

        # ── 多球心（不参与梯度）──
        self.register_buffer(
            "centers",
            torch.zeros(self.num_classes, self.feature_dim)
        )
        self.centers_initialized = False

        # ── 每类独立阈值（推理用，由评估阶段设定）──
        self.register_buffer(
            "thresholds",
            torch.full((self.num_classes,), float("inf"))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, C, ...)
        Returns:
            feat: 特征向量 (B, feature_dim)，若 normalize_feat=True 则为单位向量
        """
        feat = self.backbone(x)
        if self.normalize_feat:
            feat = F.normalize(feat, p=2, dim=1, eps=1e-8)
        return feat

    def compute_distances(self, feat: torch.Tensor) -> torch.Tensor:
        """
        计算每个样本到所有球心的欧氏距离平方。

        Args:
            feat: (B, feature_dim)
        Returns:
            dist_sq: (B, K) 到每个球心的距离平方
        """
        # (B, 1, D) - (1, K, D) -> (B, K, D) -> sum -> (B, K)
        diff = feat.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return dist_sq

    def compute_own_and_others(self, feat: torch.Tensor, labels: torch.Tensor):
        """
        辅助函数：拆分出“自身类别距离”和“完整距离矩阵”，方便计算损失。

        Args:
            feat:   (B, feature_dim)
            labels: (B,) 取值范围 [0, K-1]，代表已知类别标签
        Returns:
            dist_sq_own: (B,) 每个样本到自身类别球心的距离平方
            dist_sq_all: (B, K) 完整距离矩阵
        """
        dist_sq_all = self.compute_distances(feat)
        dist_sq_own = dist_sq_all.gather(1, labels.view(-1, 1)).squeeze(1)
        return dist_sq_own, dist_sq_all

    @torch.no_grad()
    def init_centers(self, data_loader, device: torch.device, eps: float = 0.1):
        """
        用各已知类别样本的特征均值分别初始化各自球心。
        调用时机: warm-up 阶段结束后。

        要求 data_loader 产出 (x, labels, ...)，labels 取值 [0, K-1]。
        """
        self.eval()
        feats_per_class = {c: [] for c in range(self.num_classes)}

        for batch in data_loader:
            x, labels = batch[0], batch[1]
            x = x.to(device)
            feat = self.forward(x)
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    feats_per_class[c].append(feat[mask].cpu())

        centers = torch.zeros(self.num_classes, self.feature_dim)
        for c in range(self.num_classes):
            if len(feats_per_class[c]) == 0:
                raise RuntimeError(f"No samples found for class {c} during center init!")
            all_feats = torch.cat(feats_per_class[c], dim=0)
            center = all_feats.mean(dim=0)

            # 防止球心坍缩到 0 附近
            center[(center.abs() < eps) & (center < 0)] = -eps
            center[(center.abs() < eps) & (center >= 0)] = eps

            if self.normalize_feat:
                center = F.normalize(center, p=2, dim=0, eps=1e-8)

            centers[c] = center

        self.centers = centers.to(device)
        self.centers_initialized = True
        self.train()
        print(f"[Multi-SVDD] Centers initialized | "
              f"norms={self.centers.norm(dim=1).tolist()}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        单批推理：返回特征、到各球心距离、预测类别、是否已知类判定。

        Returns:
            dict with keys:
              "feat"       : (B, feature_dim)
              "dist"       : (B, K) 到每个球心的欧氏距离
              "min_dist"   : (B,) 到最近球心的距离
              "pred_class" : (B,) 预测类别，-1 表示判定为未知类（开集拒识）
              "is_known"   : (B,) bool, True → 判定为已知类中某一类
        """
        self.eval()
        feat = self.forward(x)
        dist_sq = self.compute_distances(feat)
        dist = dist_sq.sqrt()

        min_dist, nearest_class = dist.min(dim=1)
        class_threshold = self.thresholds[nearest_class]
        is_known = min_dist < class_threshold

        pred_class = torch.where(
            is_known, nearest_class,
            torch.full_like(nearest_class, -1)
        )

        return {
            "feat": feat,
            "dist": dist,
            "min_dist": min_dist,
            "pred_class": pred_class,
            "is_known": is_known
        }


# 向后兼容别名（如旧代码引用 DeepSVDDModel，可通过 num_classes=1 退化为单球体）
DeepSVDDModel = MultiSphereDeepSVDDModel


# ─────────────────────────────────────────────
#  损失函数
# ─────────────────────────────────────────────

class MultiSphereSVDDLoss(nn.Module):
    """
    多球体 Deep SVDD 损失:

      L = mean(dist_sq_own)                                # 自身类别聚拢到对应球心
        + lambda_neg * mean(ReLU(margin - dist_sq_other))   # 其他球心排斥该样本

    说明:
      - dist_sq_own:   每个样本到自身类别球心的距离平方，越小越好
      - dist_sq_other: 每个样本到非自身类别球心的距离平方矩阵，应大于 sqrt(margin)
      - lambda_neg:    排斥项权重
      - margin:        非自身球心距离平方的目标下界
    """

    def __init__(self, lambda_neg: float = 1.0):
        super().__init__()
        self.lambda_neg = lambda_neg

    def forward(self,
                dist_sq_all: torch.Tensor,
                labels: torch.Tensor,
                margin: float = 4.0) -> torch.Tensor:
        """
        Args:
            dist_sq_all: (B, K) 到所有球心的距离平方，K为已知类别数
            labels:      (B,) 取值 [0, K-1] 为已知类别；
                         若存在开集负样本(不属于任何已知类)，标记为 -1，
                         其对所有球心都视为“他类”参与排斥项，不参与聚拢项
            margin:      排斥项目标下界
        """
        B, K = dist_sq_all.shape
        device = dist_sq_all.device
        idx = torch.arange(B, device=device)

        known_mask = labels >= 0

        # ── 聚拢项：只对已知类样本计算 ──
        if known_mask.sum() > 0:
            own_dist = dist_sq_all[idx[known_mask], labels[known_mask]]
            loss_pos = own_dist.mean()
        else:
            loss_pos = torch.tensor(0.0, device=device)

        # ── 排斥项：对“非自身类别”的所有球心计算 ──
        mask_other = torch.ones_like(dist_sq_all, dtype=torch.bool)
        if known_mask.sum() > 0:
            mask_other[idx[known_mask], labels[known_mask]] = False
        # 未知/负样本（label=-1）对所有K个球心都参与排斥，故不做屏蔽

        other_dist = dist_sq_all[mask_other]
        if other_dist.numel() > 0:
            loss_neg = F.relu(margin - other_dist).mean()
        else:
            loss_neg = torch.tensor(0.0, device=device)

        return loss_pos + self.lambda_neg * loss_neg


class CenterSeparationLoss(nn.Module):
    """
    球心分离损失：防止多个球心互相靠近甚至重合导致类别不可分。

      L = mean(ReLU(margin_center - ||c_i - c_j||^2)), i != j
    """

    def __init__(self, margin_center: float = 4.0):
        super().__init__()
        self.margin_center = margin_center

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centers: (K, D)
        """
        K = centers.size(0)
        if K < 2:
            return torch.tensor(0.0, device=centers.device)

        dist_sq = torch.cdist(centers, centers, p=2) ** 2  # (K, K)
        eye = torch.eye(K, dtype=torch.bool, device=centers.device)
        off_diag = dist_sq[~eye]

        return F.relu(self.margin_center - off_diag).mean()


class ContrastiveLoss(nn.Module):
    """
    简单对比损失（作为辅助损失，多分类版）：
      同类样本对: 距离应小
      异类样本对: 距离应大于 margin
    labels 现为多分类整数（不再局限于 0/1）。
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = feat.size(0)
        if B < 2:
            return torch.tensor(0.0, device=feat.device)

        dist_mat = torch.cdist(feat, feat, p=2)

        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))
        neg_mask = ~pos_mask
        eye = torch.eye(B, dtype=torch.bool, device=feat.device)
        pos_mask = pos_mask & ~eye
        neg_mask = neg_mask & ~eye

        loss = torch.tensor(0.0, device=feat.device)
        if pos_mask.sum() > 0:
            loss = loss + dist_mat[pos_mask].mean()
        if neg_mask.sum() > 0:
            loss = loss + F.relu(self.margin - dist_mat[neg_mask]).mean()

        return loss


# ─────────────────────────────────────────────
#  快速测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("E:/XieYiR/UAV_osr/configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 假设 cfg["model"]["num_classes"] = 4  (多类已知信号)
    model = MultiSphereDeepSVDDModel(cfg)
    print(model)

    batch = torch.randn(8, 1, 1024)          # 原始电压序列
    labels = torch.randint(0, model.num_classes, (8,))
    print("输入形状: ", batch.shape)

    feat = model(batch)
    print(f"特征形状: {feat.shape}  (应为单位范数向量)")
    print("特征范数: ", feat.norm(dim=1))

    dist_sq_all = model.compute_distances(feat)
    print(f"距离矩阵形状: {dist_sq_all.shape}")  # (8, K)

    svdd_loss = MultiSphereSVDDLoss(lambda_neg=1.0)
    loss = svdd_loss(dist_sq_all, labels, margin=4.0)
    print(f"Multi-Sphere SVDD loss: {loss.item():.4f}")

    sep_loss = CenterSeparationLoss(margin_center=4.0)
    loss_sep = sep_loss(model.centers)
    print(f"Center separation loss: {loss_sep.item():.4f}")

    print("model.py 多球体测试通过 ✓")