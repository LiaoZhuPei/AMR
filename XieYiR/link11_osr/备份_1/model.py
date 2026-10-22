"""
model.py
模型定义模块

包含:
  - CNN2DBackbone: 2D-CNN 特征提取器（用于时频图）
  - CNN1DBackbone: 1D-CNN 特征提取器（直接处理原始IQ/信号）
  - DeepSVDDModel: 完整模型（骨干网络 + 投影头 + 球心参数）

Deep SVDD 核心思想:
  训练时将正类样本映射到特征空间中一个紧凑的超球体内。
  推理时计算到球心的距离：距离 > 阈值 → 拒识为未知信号。
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
        """
        Args:
            x: (B, 1, freq_bins, time_frames)
        Returns:
            feat: (B, feature_dim)
        """
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


# ─────────────────────────────────────────────
#  1D-CNN 骨干（直接处理原始信号）
# ─────────────────────────────────────────────

class CNN1DBackbone(nn.Module):
    """
    1D-CNN，输入: (B, 1, signal_length)
    适合直接处理原始 IQ 或时域信号。
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
        prev_ch = 1
        for ch, ks, ps in zip(channels, kernel_sizes, pool_sizes):
            layers += [
                nn.Conv1d(prev_ch, ch, kernel_size=ks,
                          padding=ks // 2),
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
        """
        Args:
            x: (B, 1, signal_length)
        Returns:
            feat: (B, feature_dim)
        """
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x


# ─────────────────────────────────────────────
#  Deep SVDD 完整模型
# ─────────────────────────────────────────────

class DeepSVDDModel(nn.Module):
    """
    Deep SVDD 开放集识别模型。

    组件:
      1. 骨干网络 (CNN2D 或 CNN1D)：提取判别特征
      2. 可学习球心 `center` (不参与梯度下降，训练初期计算后固定)
      3. 推理时输出特征向量及到球心的距离

    训练策略:
      Phase 1 (warm-up): 仅用正类，让网络先找到合理的特征空间
      Phase 2:           SVDD 损失 + 辅助对比损失同步优化
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        self.feature_dim = model_cfg["feature_dim"]
        self.backbone_type = model_cfg["backbone"]

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

        # ── 球心（不参与梯度）──
        self.register_buffer(
            "center",
            torch.zeros(self.feature_dim)
        )
        self.center_initialized = False

        # ── 阈值（推理用，由评估阶段设定）──
        self.register_buffer(
            "threshold",
            torch.tensor(float("inf"))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, C, ...)
        Returns:
            feat: 特征向量 (B, feature_dim)
        """
        feat = self.backbone(x)
        return feat

    def compute_distances(self, feat: torch.Tensor) -> torch.Tensor:
        """
        计算每个样本到超球心的欧氏距离平方。

        Args:
            feat: (B, feature_dim)
        Returns:
            dist_sq: (B,) 距离平方
        """
        diff = feat - self.center.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=1)
        return dist_sq

    @torch.no_grad()
    def init_center(self, data_loader, device: torch.device,
                    eps: float = 0.1):
        """
        用正类样本的特征均值初始化球心。
        调用时机: warm-up 阶段结束后。

        Args:
            data_loader: 仅含正类样本的 DataLoader
            device:      计算设备
            eps:         防止球心坍缩到 0 附近的最小值
        """
        self.eval()
        all_feats = []
        for x, labels, _ in data_loader:
            x = x.to(device)
            mask = (labels == 1)
            if mask.sum() == 0:
                continue
            feat = self.forward(x[mask])
            all_feats.append(feat.cpu())

        if not all_feats:
            raise RuntimeError("No positive samples found for center init!")

        all_feats = torch.cat(all_feats, dim=0)
        center = all_feats.mean(dim=0)

        # 防止球心坍缩
        center[(center.abs() < eps) & (center < 0)] = -eps
        center[(center.abs() < eps) & (center >= 0)] = eps

        self.center = center.to(device)
        self.center_initialized = True
        self.train()
        print(f"[SVDD] Center initialized | "
              f"norm={self.center.norm().item():.4f}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        单批推理：返回特征、距离、决策。

        Returns:
            dict with keys:
              "feat"      : (B, feature_dim)
              "dist"      : (B,) 欧氏距离
              "is_target" : (B,) bool, True → 判定为 Link11
        """
        self.eval()
        feat = self.forward(x)
        dist_sq = self.compute_distances(feat)
        dist = dist_sq.sqrt()
        is_target = dist < self.threshold
        return {
            "feat": feat,
            "dist": dist,
            "is_target": is_target
        }


# ─────────────────────────────────────────────
#  损失函数
# ─────────────────────────────────────────────

class SVDDLoss(nn.Module):
    """
    Deep SVDD 损失：
      L = (1/n) * Σ max(0, dist_sq - R²) + (1/(nu*n)) * Σ [if outlier]

    简化版（软边界）：
      L_pos = mean(dist_sq_pos)         ← 正类尽量靠近球心
      L_neg = mean(max(0, m - dist_neg)) ← 负类尽量远离球心 (margin m)
    """

    def __init__(self, nu: float = 0.1):
        super().__init__()
        self.nu = nu

    def forward(self,
                dist_sq_pos: torch.Tensor,
                dist_sq_neg: torch.Tensor = None,
                margin: float = 4.0) -> torch.Tensor:
        """
        Args:
            dist_sq_pos: 正类距离平方 (N_pos,)
            dist_sq_neg: 负类距离平方 (N_neg,), 可为 None
            margin:      负类排斥边界 (距离平方)
        """
        loss_pos = dist_sq_pos.mean()

        if dist_sq_neg is not None and len(dist_sq_neg) > 0:
            # 负类应远离球心：penalty when dist_sq_neg < margin
            loss_neg = F.relu(margin - dist_sq_neg).mean()
            return loss_pos + (1.0 / self.nu) * loss_neg * 0.1
        return loss_pos


class ContrastiveLoss(nn.Module):
    """
    简单对比损失（作为辅助损失）：
      正对 (都是正类): 距离应小
      负对 (一正一负): 距离应大于 margin
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(self,
                feat: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:   (B, D) 特征向量
            labels: (B,) 0或1
        """
        B = feat.size(0)
        if B < 2:
            return torch.tensor(0.0, device=feat.device)

        # 计算成对距离
        dist_mat = torch.cdist(feat, feat, p=2)  # (B, B)

        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0))   # 同类
        neg_mask = ~pos_mask
        # 去除对角线
        eye = torch.eye(B, dtype=torch.bool, device=feat.device)
        pos_mask = pos_mask & ~eye
        neg_mask = neg_mask & ~eye

        loss = torch.tensor(0.0, device=feat.device)
        if pos_mask.sum() > 0:
            loss += dist_mat[pos_mask].mean()
        if neg_mask.sum() > 0:
            loss += F.relu(self.margin - dist_mat[neg_mask]).mean()

        return loss


# ─────────────────────────────────────────────
#  快速测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = DeepSVDDModel(cfg)
    print(model)

    # 模拟一批频谱图输入
    batch = torch.randn(8, 1, 129, 128)
    feat = model(batch)
    print(f"Feature shape: {feat.shape}")  # (8, 128)

    # 测试损失
    svdd_loss = SVDDLoss(nu=0.1)
    dist_sq = model.compute_distances(feat)
    print(f"Distance sq shape: {dist_sq.shape}")
    loss = svdd_loss(dist_sq)
    print(f"SVDD loss: {loss.item():.4f}")
    print("model.py 测试通过 ✓")
