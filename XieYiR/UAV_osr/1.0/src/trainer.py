"""
trainer.py
训练逻辑模块（多球体 Deep SVDD 版本）

训练流程:
  1. Warm-up 阶段: 用已知类样本做多分类预训练（临时分类头），
                    让骨干网络先学到可分的簇结构
  2. 球心初始化: 对每个已知类别分别计算特征均值作为该类球心
  3. Multi-Sphere SVDD 训练: 优化 "自身类聚拢 + 他类球心排斥" 损失，
                    可选周期性 EMA 更新球心以维持类间可分性
  4. 阈值标定: 在验证集上按类别统计距离分布，标定每类拒识阈值
  5. 早停 & 检查点: 按验证集综合指标（闭集准确率 + 开集AUROC）保存最优模型
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.model import (
    MultiSphereDeepSVDDModel,
    MultiSphereSVDDLoss,
    CenterSeparationLoss,
    ContrastiveLoss
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    多球体 Deep SVDD 训练器。

    Args:
        model:        MultiSphereDeepSVDDModel 实例
        train_loader: 训练 DataLoader，产出 (x, labels, meta)，
                      labels ∈ {0,...,K-1}（已知类），可选 -1（开集/未知类，用于训练时的排斥项）
        val_loader:   验证 DataLoader，同样格式，可包含 -1 标签用于评估开集检测能力
        cfg:          完整配置字典
        device:       torch.device
    """

    def __init__(self,
                 model: MultiSphereDeepSVDDModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 cfg: dict,
                 device: torch.device):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.num_classes = model.num_classes

        train_cfg = cfg["training"]
        svdd_cfg = train_cfg["svdd"]

        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.warm_up_epochs = svdd_cfg["warm_up_epochs"]
        self.patience = train_cfg["patience"]
        self.log_interval = cfg["output"]["log_interval"]

        self.use_contrastive = train_cfg.get("use_aux_contrastive", True)
        self.lambda_contrastive = train_cfg.get("lambda_contrastive", 0.3)
        contrastive_margin = train_cfg.get("contrastive_margin", 2.0)

        # ── 球心分离 / 更新相关配置 ──
        self.lambda_center_sep = svdd_cfg.get("lambda_center_sep", 0.0)
        self.margin_center = svdd_cfg.get("margin_center", 4.0)
        # centers 是 buffer（不参与梯度），故分离损失仅用于监控；
        # 若想让分离约束真正生效，需周期性用 EMA 重新估计 centers
        self.update_centers_every = svdd_cfg.get("update_centers_every", 0)  # 0 表示不更新
        self.center_momentum = svdd_cfg.get("center_momentum", 0.9)

        # ── 阈值标定配置 ──
        self.threshold_percentile = svdd_cfg.get("threshold_percentile", 0.95)

        # 损失函数
        lambda_neg = svdd_cfg.get("lambda_neg", 1.0)
        self.svdd_margin = svdd_cfg.get("margin", 4.0)
        self.svdd_loss_fn = MultiSphereSVDDLoss(lambda_neg=lambda_neg)
        self.center_sep_loss_fn = CenterSeparationLoss(margin_center=self.margin_center)
        self.contrastive_loss_fn = ContrastiveLoss(margin=contrastive_margin)

        # ── Warm-up 专用临时分类头（热身结束后丢弃）──
        self.warmup_head = nn.Linear(self.model.feature_dim, self.num_classes).to(device)

        # ── 优化器 ──
        # warm-up 阶段: 骨干网络 + 临时分类头 一起训练
        self.warmup_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.warmup_head.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # SVDD 阶段: 只训练骨干网络（分类头已丢弃）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 学习率调度（仅作用于 SVDD 阶段的 optimizer）
        scheduler_type = train_cfg.get("lr_scheduler", "cosine")
        svdd_epochs = max(self.epochs - self.warm_up_epochs, 1)
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=svdd_epochs
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        else:
            self.scheduler = None

        # 训练状态
        self.best_score: float = -1.0   # 综合指标（闭集acc + 开集auroc）
        self.no_improve_count = 0
        self.model_dir = cfg["output"]["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    #  Warm-up 阶段（多分类预训练）
    # ─────────────────────────────────────────────

    def _warmup_epoch(self, epoch: int):
        """
        热身阶段: 用已知类样本做多分类交叉熵预训练。

        目的: 让骨干网络在进入 Multi-Sphere SVDD 训练前，
              已经学到 K 个类别的基本判别方向，
              使得后续球心初始化时各簇彼此分离、结构合理。

        损失: CrossEntropy(warmup_head(feat), label)
              warmup_head 是临时线性分类头，热身结束后丢弃。
        """
        self.model.train()
        self.warmup_head.train()
        total_loss = 0.0
        n_correct = 0
        n_total = 0
        n_batches = 0

        for batch_idx, (x, labels, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            labels = labels.to(self.device).long()

            # 仅用已知类样本做多分类预训练，过滤掉开集负样本(-1)
            known_mask = labels >= 0
            if known_mask.sum() == 0:
                continue
            x = x[known_mask]
            labels = labels[known_mask]

            feat = self.model(x)                  # (B, D) 已归一化
            logits = self.warmup_head(feat)        # (B, K)
            loss = F.cross_entropy(logits, labels)

            self.warmup_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.warmup_head.parameters()), 1.0
            )
            self.warmup_optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_total += labels.size(0)

        avg_loss = total_loss / max(n_batches, 1)
        acc = n_correct / max(n_total, 1)
        logger.info(f"[Warm-up] Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
        return avg_loss

    # ─────────────────────────────────────────────
    #  SVDD 训练阶段
    # ─────────────────────────────────────────────

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # 用于本 epoch 结束后可能的 EMA 球心更新
        feats_per_class = {c: [] for c in range(self.num_classes)}

        for batch_idx, (x, labels, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            labels = labels.to(self.device).long()

            feat = self.model(x)                          # (B, D)
            dist_sq_all = self.model.compute_distances(feat)  # (B, K)

            # ── Multi-Sphere SVDD 损失 ──
            loss = self.svdd_loss_fn(dist_sq_all, labels, margin=self.svdd_margin)

            # ── 球心分离监控项（centers 是 buffer，不参与梯度，仅用于日志监控）──
            with torch.no_grad():
                sep_val = self.center_sep_loss_fn(self.model.centers).item()

            # ── 对比辅助损失（仅用已知类样本）──
            known_mask = labels >= 0
            if self.use_contrastive and known_mask.sum() > 1:
                c_loss = self.contrastive_loss_fn(feat[known_mask], labels[known_mask])
                loss = loss + self.lambda_contrastive * c_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # 收集特征用于周期性 EMA 更新球心
            if self.update_centers_every > 0:
                with torch.no_grad():
                    for c in range(self.num_classes):
                        mask_c = (labels == c)
                        if mask_c.sum() > 0:
                            feats_per_class[c].append(feat[mask_c].detach().cpu())

            if (batch_idx + 1) % self.log_interval == 0:
                logger.debug(
                    f"Epoch {epoch:03d} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | CenterSep: {sep_val:.4f}"
                )

        avg_loss = total_loss / max(n_batches, 1)

        # ── 周期性 EMA 更新球心（可选）──
        if self.update_centers_every > 0 and epoch % self.update_centers_every == 0:
            self._ema_update_centers(feats_per_class)

        return avg_loss

    @torch.no_grad()
    def _ema_update_centers(self, feats_per_class: dict):
        """
        用本 epoch 收集到的特征，以 EMA 方式更新球心，
        避免球心长期固定导致的漂移滞后，同时维持一定的类间可分性
        （比每步梯度更新球心更稳定，避免坍缩）。
        """
        updated = []
        for c in range(self.num_classes):
            if len(feats_per_class[c]) == 0:
                updated.append(self.model.centers[c].cpu())
                continue
            new_mean = torch.cat(feats_per_class[c], dim=0).mean(dim=0)
            if self.model.normalize_feat:
                new_mean = F.normalize(new_mean, dim=0, eps=1e-8)
            old_center = self.model.centers[c].cpu()
            merged = self.center_momentum * old_center + (1 - self.center_momentum) * new_mean
            if self.model.normalize_feat:
                merged = F.normalize(merged, dim=0, eps=1e-8)
            updated.append(merged)

        self.model.centers = torch.stack(updated, dim=0).to(self.device)
        sep = self.center_sep_loss_fn(self.model.centers).item()
        logger.info(f"[Centers] EMA 更新完成 | CenterSepLoss={sep:.4f}")

    # ─────────────────────────────────────────────
    #  验证
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> dict:
        """
        在验证集上评估两方面能力:
          1. 闭集多分类准确率: 已知类样本 (label >= 0) 是否被分到正确球心
          2. 开集检测 AUROC: 已知类 vs 未知类 (label == -1)，
             用"到最近球心的距离"作为异常分数（距离越大越像未知）

        Returns:
            dict: {
                "closed_acc": float,
                "open_auroc": float or None,
                "score": float  # 综合指标，用于早停/保存判断
            }
        """
        self.model.eval()
        all_min_dist = []
        all_nearest_class = []
        all_labels = []

        for x, labels, _ in self.val_loader:
            x = x.to(self.device)
            feat = self.model(x)
            dist_sq_all = self.model.compute_distances(feat)
            dist_all = dist_sq_all.sqrt()

            min_dist, nearest_class = dist_all.min(dim=1)

            all_min_dist.extend(min_dist.cpu().numpy())
            all_nearest_class.extend(nearest_class.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_min_dist = np.array(all_min_dist, dtype=np.float32)
        all_nearest_class = np.array(all_nearest_class, dtype=np.int64)
        all_labels = np.array(all_labels, dtype=np.int64)

        known_mask = all_labels >= 0
        unknown_mask = all_labels == -1

        # ── 闭集多分类准确率 ──
        if known_mask.sum() > 0:
            closed_acc = float(
                (all_nearest_class[known_mask] == all_labels[known_mask]).mean()
            )
        else:
            closed_acc = 0.0

        # ── 开集检测 AUROC（已知 vs 未知）──
        open_auroc = None
        if known_mask.sum() > 0 and unknown_mask.sum() > 0:
            # 二值标签: 1=已知(正), 0=未知(负)；得分越高越像已知 → 用 -min_dist
            binary_labels = known_mask.astype(np.int32)
            open_auroc = float(roc_auc_score(binary_labels, -all_min_dist))

        # ── 诊断信息：各已知类到自身球心的平均距离 ──
        diag_msgs = []
        for c in range(self.num_classes):
            mask_c = (all_labels == c)
            if mask_c.sum() > 0:
                diag_msgs.append(f"class{c}={all_min_dist[mask_c].mean():.4f}")
        if unknown_mask.sum() > 0:
            diag_msgs.append(f"unknown={all_min_dist[unknown_mask].mean():.4f}")
        logger.info(f"[Val] dist_mean: {' | '.join(diag_msgs)}")

        # ── 综合指标 ──
        if open_auroc is not None:
            score = 0.5 * closed_acc + 0.5 * open_auroc
        else:
            score = closed_acc

        return {
            "closed_acc": closed_acc,
            "open_auroc": open_auroc,
            "score": score
        }

    @torch.no_grad()
    def _calibrate_thresholds(self):
        """
        在验证集上按类别标定拒识阈值：
          对每个已知类别 c，取该类样本到自身球心的距离分布，
          用指定分位数 (如 95%) 作为该类阈值，
          使得约 95% 的该类正常样本可以被正确接受。
        """
        self.model.eval()
        dists_per_class = {c: [] for c in range(self.num_classes)}

        for x, labels, _ in self.val_loader:
            x = x.to(self.device)
            labels = labels.to(self.device).long()
            feat = self.model(x)
            dist_sq_all = self.model.compute_distances(feat)
            dist_all = dist_sq_all.sqrt()

            for c in range(self.num_classes):
                mask_c = (labels == c)
                if mask_c.sum() > 0:
                    dists_per_class[c].append(dist_all[mask_c, c].cpu())

        thresholds = torch.full((self.num_classes,), float("inf"))
        for c in range(self.num_classes):
            if len(dists_per_class[c]) == 0:
                continue
            all_d = torch.cat(dists_per_class[c])
            thresholds[c] = torch.quantile(all_d, self.threshold_percentile)

        self.model.thresholds = thresholds.to(self.device)
        logger.info(f"[Threshold] 标定完成 | thresholds={self.model.thresholds.tolist()}")

    # ─────────────────────────────────────────────
    #  主训练循环
    # ─────────────────────────────────────────────

    def train(self):
        """执行完整训练流程"""
        print(f"\n{'='*60}")
        print(f"  开始训练 (Multi-Sphere SVDD) | 设备: {self.device}")
        print(f"  已知类别数 K={self.num_classes}")
        print(f"  总 Epochs: {self.epochs} | Warm-up: {self.warm_up_epochs}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):

            # ── Phase 1: Warm-up ──
            if epoch <= self.warm_up_epochs:
                loss = self._warmup_epoch(epoch)
                if epoch == self.warm_up_epochs:
                    print(f"\n[Trainer] Warm-up 完成，初始化多球心...")
                    self.model.init_centers(self.train_loader, self.device)
                    del self.warmup_head  # 丢弃临时分类头
                continue

            # ── Phase 2: Multi-Sphere SVDD 训练 ──
            train_loss = self._train_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            # 验证 + 阈值标定
            self._calibrate_thresholds()
            val_result = self._validate()

            elapsed = time.time() - start_time
            open_auroc_str = (f"{val_result['open_auroc']:.4f}"
                               if val_result["open_auroc"] is not None else "N/A")
            print(f"Epoch {epoch:03d}/{self.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"ClosedAcc: {val_result['closed_acc']:.4f} | "
                  f"OpenAUROC: {open_auroc_str} | "
                  f"Score: {val_result['score']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {elapsed:.1f}s")

            # ── 保存最优 ──
            score = val_result["score"]
            if score > self.best_score:
                self.best_score = score
                self.no_improve_count = 0
                self._save_checkpoint(epoch, val_result, is_best=True)
            else:
                self.no_improve_count += 1

            # ── 早停 ──
            if self.no_improve_count >= self.patience:
                print(f"\n[早停] {self.patience} 轮无提升，停止训练。"
                      f"最佳 Score: {self.best_score:.4f}")
                break

        print(f"\n{'='*60}")
        print(f"  训练完成 | 最佳 Score: {self.best_score:.4f}")
        print(f"{'='*60}\n")

    def _save_checkpoint(self, epoch: int, val_result: dict,
                         is_best: bool = False):
        """保存模型检查点"""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_result": val_result,
            "cfg": self.cfg
        }
        path = os.path.join(self.model_dir,
                            "best_model.pth" if is_best
                            else f"ckpt_epoch{epoch:03d}.pth")
        torch.save(state, path)
        if is_best:
            logger.info(f"[Checkpoint] 最优模型已保存 → {path} "
                        f"(Score={val_result['score']:.4f}, "
                        f"ClosedAcc={val_result['closed_acc']:.4f}, "
                        f"OpenAUROC={val_result['open_auroc']})")

    def load_best_model(self):
        """加载最优检查点"""
        path = os.path.join(self.model_dir, "best_model.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        vr = state["val_result"]
        print(f"[Trainer] 最优模型已加载 (Epoch {state['epoch']}, "
              f"Score={vr['score']:.4f}, ClosedAcc={vr['closed_acc']:.4f}, "
              f"OpenAUROC={vr['open_auroc']})")
        return state