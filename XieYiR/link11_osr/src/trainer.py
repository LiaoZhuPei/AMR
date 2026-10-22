"""
trainer.py
训练逻辑模块

训练流程:
  1. Warm-up 阶段: 仅用正类样本，用简单 MSE 损失让网络先学习正类特征
  2. 球心初始化: 计算所有正类样本特征的均值作为球心
  3. SVDD 训练: 优化 SVDD 损失（正类聚拢 + 负类排斥）+ 可选对比损失
  4. 早停 & 检查点: 按验证集 AUROC 保存最优模型
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from src.model import DeepSVDDModel, SVDDLoss, ContrastiveLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Deep SVDD 训练器。

    Args:
        model:        DeepSVDDModel 实例
        train_loader: 训练 DataLoader
        val_loader:   验证 DataLoader
        cfg:          完整配置字典
        device:       torch.device
    """

    def __init__(self,
                 model: DeepSVDDModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 cfg: dict,
                 device: torch.device):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        train_cfg = cfg["training"]
        svdd_cfg = train_cfg["svdd"]

        self.epochs = train_cfg["epochs"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]
        self.warm_up_epochs = svdd_cfg["warm_up_epochs"]
        self.nu = svdd_cfg["nu"]
        self.patience = train_cfg["patience"]
        self.log_interval = cfg["output"]["log_interval"]

        self.use_contrastive = train_cfg.get("use_aux_contrastive", True)
        self.lambda_contrastive = train_cfg.get("lambda_contrastive", 0.3)
        contrastive_margin = train_cfg.get("contrastive_margin", 2.0)

        # 损失函数
        lambda_neg = svdd_cfg.get("lambda_neg", 1.0)
        self.svdd_margin = svdd_cfg.get("margin", 4.0)
        self.svdd_loss_fn = SVDDLoss(nu=self.nu, lambda_neg=lambda_neg)
        self.contrastive_loss_fn = ContrastiveLoss(margin=contrastive_margin)

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 学习率调度
        scheduler_type = train_cfg.get("lr_scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        else:
            self.scheduler = None

        # 训练状态
        self.best_auroc: float = -1.0
        self.no_improve_count = 0
        self.model_dir = cfg["output"]["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    #  Warm-up 阶段
    # ─────────────────────────────────────────────

    def _warmup_epoch(self, epoch: int):
        """
        热身阶段: 用正类 + 已知负类做二分类预训练。

        目的: 让骨干网络在进入 SVDD 训练前，已经学到
              "正类 vs 负类"的基本判别方向，避免球心
              落在无意义的零点附近。

        损失: BCE(sigmoid(w·feat), label)
              w 是一个临时线性头，热身结束后丢弃。
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, labels, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            labels = labels.to(self.device).float()

            feat = self.model(x)           # (B, D)

            # 临时线性得分: 用特征 L2 范数作为正类得分的代理
            # 正类应有更集中/更大的激活 → 用特征均值投影
            # 实际做法: 每 batch 内对正类特征取均值作为"原型"，
            # 用余弦相似度做二分类
            pos_mask = (labels == 1)
            if pos_mask.sum() == 0:
                continue

            # 正类原型（当前 batch）
            proto = feat[pos_mask].mean(dim=0, keepdim=True)  # (1, D)
            proto = nn.functional.normalize(proto, dim=1)

            feat_norm = nn.functional.normalize(feat, dim=1)  # (B, D)
            logits = (feat_norm * proto).sum(dim=1)            # (B,) cos sim

            # BCE：正类→1，负类→0
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"[Warm-up] Epoch {epoch:03d} | Loss: {avg_loss:.4f}")
        return avg_loss

    # ─────────────────────────────────────────────
    #  SVDD 训练阶段
    # ─────────────────────────────────────────────

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, labels, _) in enumerate(self.train_loader):
            x = x.to(self.device)
            labels = labels.to(self.device)

            feat = self.model(x)
            dist_sq = self.model.compute_distances(feat)

            pos_mask = (labels == 1)
            neg_mask = (labels == 0)

            # ── SVDD 损失 ──
            if pos_mask.sum() == 0:
                continue
            dist_sq_pos = dist_sq[pos_mask]
            dist_sq_neg = dist_sq[neg_mask] if neg_mask.sum() > 0 else None

            loss = self.svdd_loss_fn(dist_sq_pos, dist_sq_neg, margin=self.svdd_margin)

            # ── 对比辅助损失 ──
            if self.use_contrastive:
                c_loss = self.contrastive_loss_fn(feat, labels)
                loss = loss + self.lambda_contrastive * c_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                logger.debug(
                    f"Epoch {epoch:03d} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    # ─────────────────────────────────────────────
    #  验证
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> float:
        """
        在验证集上计算 AUROC，并打印正负类距离均值用于诊断。

        约定:
          距离小 → 靠近球心 → 正类(Link11)
          距离大 → 远离球心 → 负类/未知

        roc_auc_score 期望"得分越高越像正类"，
        因此传入 -dist（距离越小 → 负距离越大 → 更像正类）。
        """
        self.model.eval()
        all_dists  = []
        all_labels = []

        for x, labels, _ in self.val_loader:
            x = x.to(self.device)
            feat    = self.model(x)
            dist_sq = self.model.compute_distances(feat)
            dist    = dist_sq.sqrt()

            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_dists  = np.array(all_dists,  dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int32)

        if len(np.unique(all_labels)) < 2:
            return 0.5

        # 诊断信息：正负类距离均值，用于判断球是否在收紧
        pos_dist_mean = all_dists[all_labels == 1].mean()
        neg_dist_mean = all_dists[all_labels == 0].mean()
        logger.info(
            f"[Val] dist_mean: pos={pos_dist_mean:.4f}  neg={neg_dist_mean:.4f}  "
            f"gap={neg_dist_mean - pos_dist_mean:.4f}"
        )

        # 正类得分 = -dist（越小越像正类）
        auroc = roc_auc_score(all_labels, -all_dists)
        return float(auroc)

    # ─────────────────────────────────────────────
    #  主训练循环
    # ─────────────────────────────────────────────

    def train(self):
        """执行完整训练流程"""
        print(f"\n{'='*60}")
        print(f"  开始训练 | 设备: {self.device}")
        print(f"  总 Epochs: {self.epochs} | Warm-up: {self.warm_up_epochs}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):

            # ── Phase 1: Warm-up ──
            if epoch <= self.warm_up_epochs:
                loss = self._warmup_epoch(epoch)
                # warm-up 结束时初始化球心
                if epoch == self.warm_up_epochs:
                    print(f"\n[Trainer] Warm-up 完成，初始化 SVDD 球心...")
                    self.model.init_center(
                        self.train_loader, self.device
                    )
                continue

            # ── Phase 2: SVDD 训练 ──
            train_loss = self._train_epoch(epoch)

            if self.scheduler:
                self.scheduler.step()

            # 验证
            auroc = self._validate()

            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d}/{self.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val AUROC: {auroc:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {elapsed:.1f}s")

            # ── 保存最优 ──
            auroc = float(auroc)
            if auroc > self.best_auroc:
                self.best_auroc = auroc
                self.no_improve_count = 0
                self._save_checkpoint(epoch, auroc, is_best=True)
            else:
                self.no_improve_count += 1

            # ── 早停 ──
            if self.no_improve_count >= self.patience:
                print(f"\n[早停] {self.patience} 轮无提升，停止训练。"
                      f"最佳 AUROC: {self.best_auroc:.4f}")
                break

        print(f"\n{'='*60}")
        print(f"  训练完成 | 最佳 Val AUROC: {self.best_auroc:.4f}")
        print(f"{'='*60}\n")

    def _save_checkpoint(self, epoch: int, auroc: float,
                         is_best: bool = False):
        """保存模型检查点"""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "auroc": auroc,
            "cfg": self.cfg
        }
        path = os.path.join(self.model_dir,
                            "best_model.pth" if is_best
                            else f"ckpt_epoch{epoch:03d}.pth")
        torch.save(state, path)
        if is_best:
            logger.info(f"[Checkpoint] 最优模型已保存 → {path} "
                        f"(AUROC={auroc:.4f})")

    def load_best_model(self):
        """加载最优检查点"""
        path = os.path.join(self.model_dir, "best_model.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        print(f"[Trainer] 最优模型已加载 (Epoch {state['epoch']}, "
              f"AUROC={state['auroc']:.4f})")
        return state
