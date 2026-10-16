"""
train.py
训练入口 - 所有训练相关配置和逻辑都在此文件中

用法:
  python train.py
"""

import os
import sys
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import split_train_val, build_dataloader
from src.model import (
    MultiSphereDeepSVDDModel, MultiSphereSVDDLoss,
    CenterSeparationLoss, ContrastiveLoss,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════
#  配置 (所有训练参数集中在此)
# ═════════════════════════════════════════════
CFG = {
    "data": {
        "train_npy": "dataset_npy/train_M2048.npy",
        "test_npy":  "dataset_npy/test_M2048.npy",     # evaluate.py 会用到
        "target_classes":  ["A", "D", "E", "G"], # 目标类
        "unknown_classes": ["B"],                # 训练时可见的负类; evaluate 会自动把 C/F 也当未知
        "train_ratio": 0.8,
        "random_seed": 42,
        "normalize": "zscore",
    },
    "model": {
        "backbone": "cnn1d",           # "cnn1d" 处理 IQ; "cnn2d" 处理时频图
        "feature_dim": 128,
        "num_classes": 4,              # = len(target_classes)
        "normalize_feature": True,
        "cnn1d": {
            "channels":     [32, 64, 128],
            "kernel_sizes": [7, 5, 3],
            "pool_sizes":   [2, 2, 2],
        },
        "cnn2d": {"channels": [16, 32, 64, 128]},
    },
    "training": {
        "epochs": 80,
        "batch_size": 128,
        "num_workers": 0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "patience": 15,
        "lr_scheduler": "cosine",
        "use_aux_contrastive": True,
        "lambda_contrastive": 0.3,
        "contrastive_margin": 2.0,
        "svdd": {
            "warm_up_epochs": 10,
            "margin": 4.0,
            "lambda_neg": 1.0,
            "lambda_center_sep": 0.0,
            "margin_center": 4.0,
            "update_centers_every": 5,
            "center_momentum": 0.9,
            "threshold_percentile": 0.95,
        },
    },
    "output": {
        "model_dir": "models",
        "results_dir": "results",
        "log_interval": 20,
    },
    "evaluation": {
        "threshold_strategy": "percentile",
        "percentile": 95,
    },
}


# ═════════════════════════════════════════════
#  工具函数
# ═════════════════════════════════════════════
def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU")
    return dev


# ═════════════════════════════════════════════
#  训练器
# ═════════════════════════════════════════════
class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.K = model.num_classes

        tc = cfg["training"]; sc = tc["svdd"]
        self.epochs = tc["epochs"]
        self.warm_up = sc["warm_up_epochs"]
        self.patience = tc["patience"]
        self.log_interval = cfg["output"]["log_interval"]

        self.use_contrast = tc["use_aux_contrastive"]
        self.lam_contrast = tc["lambda_contrastive"]
        self.svdd_margin = sc["margin"]
        self.update_centers_every = sc["update_centers_every"]
        self.center_mom = sc["center_momentum"]
        self.threshold_pct = sc["threshold_percentile"]

        self.svdd_loss = MultiSphereSVDDLoss(lambda_neg=sc["lambda_neg"])
        self.sep_loss = CenterSeparationLoss(margin_center=sc["margin_center"])
        self.contrast_loss = ContrastiveLoss(margin=tc["contrastive_margin"])

        self.warmup_head = nn.Linear(model.feature_dim, self.K).to(device)
        self.warmup_opt = optim.Adam(
            list(model.parameters()) + list(self.warmup_head.parameters()),
            lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        self.opt = optim.Adam(model.parameters(),
                              lr=tc["learning_rate"], weight_decay=tc["weight_decay"])

        svdd_epochs = max(self.epochs - self.warm_up, 1)
        if tc["lr_scheduler"] == "cosine":
            self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=svdd_epochs)
        elif tc["lr_scheduler"] == "step":
            self.sched = optim.lr_scheduler.StepLR(self.opt, 20, 0.5)
        else:
            self.sched = None

        self.best_score = -1.0
        self.no_improve = 0
        self.model_dir = cfg["output"]["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    # ── Warm-up ──
    def _warmup_epoch(self, epoch):
        self.model.train(); self.warmup_head.train()
        tot, correct, n = 0.0, 0, 0
        for x, y, _ in self.train_loader:
            x, y = x.to(self.device), y.to(self.device).long()
            m = y >= 0
            if not m.any(): continue
            x, y = x[m], y[m]
            feat = self.model(x)
            logits = self.warmup_head(feat)
            loss = F.cross_entropy(logits, y)
            self.warmup_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.warmup_head.parameters()), 1.0)
            self.warmup_opt.step()
            tot += loss.item()
            correct += (logits.argmax(1) == y).sum().item(); n += y.size(0)
        logger.info(f"[Warm-up] Epoch {epoch:03d} Loss={tot/len(self.train_loader):.4f} "
                    f"Acc={correct/max(n,1):.4f}")

    # ── SVDD ──
    def _train_epoch(self, epoch):
        self.model.train()
        tot = 0.0
        feats_per_cls = {c: [] for c in range(self.K)}
        for i, (x, y, _) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device).long()
            feat = self.model(x)
            dist_sq = self.model.compute_distances(feat)
            loss = self.svdd_loss(dist_sq, y, margin=self.svdd_margin)
            if self.use_contrast:
                m = y >= 0
                if m.sum() > 1:
                    loss = loss + self.lam_contrast * self.contrast_loss(feat[m], y[m])

            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            tot += loss.item()

            if self.update_centers_every > 0:
                with torch.no_grad():
                    for c in range(self.K):
                        m = y == c
                        if m.any():
                            feats_per_cls[c].append(feat[m].detach().cpu())

        if self.update_centers_every > 0 and epoch % self.update_centers_every == 0:
            self._ema_update_centers(feats_per_cls)
        return tot / len(self.train_loader)

    @torch.no_grad()
    def _ema_update_centers(self, feats):
        new = []
        for c in range(self.K):
            if feats[c]:
                mean = torch.cat(feats[c], 0).mean(0)
                if self.model.normalize_feat:
                    mean = F.normalize(mean, dim=0, eps=1e-8)
                merged = self.center_mom * self.model.centers[c].cpu() + \
                         (1 - self.center_mom) * mean
                if self.model.normalize_feat:
                    merged = F.normalize(merged, dim=0, eps=1e-8)
                new.append(merged)
            else:
                new.append(self.model.centers[c].cpu())
        self.model.centers = torch.stack(new).to(self.device)

    # ── 验证 & 阈值 ──
    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        all_min, all_pred, all_lab = [], [], []
        for x, y, _ in self.val_loader:
            x = x.to(self.device)
            d = self.model.compute_distances(self.model(x)).sqrt()
            md, nc = d.min(1)
            all_min.append(md.cpu().numpy())
            all_pred.append(nc.cpu().numpy())
            all_lab.append(y.numpy())
        md = np.concatenate(all_min); nc = np.concatenate(all_pred); y = np.concatenate(all_lab)
        known = y >= 0; unk = y == -1
        closed_acc = float((nc[known] == y[known]).mean()) if known.any() else 0.0
        auroc = float(roc_auc_score(known.astype(int), -md)) if known.any() and unk.any() else None
        score = 0.5 * closed_acc + 0.5 * auroc if auroc is not None else closed_acc
        return {"closed_acc": closed_acc, "open_auroc": auroc, "score": score}

    @torch.no_grad()
    def _calibrate(self):
        self.model.eval()
        per_cls = {c: [] for c in range(self.K)}
        for x, y, _ in self.val_loader:
            x = x.to(self.device); y = y.to(self.device).long()
            d = self.model.compute_distances(self.model(x)).sqrt()
            for c in range(self.K):
                m = y == c
                if m.any():
                    per_cls[c].append(d[m, c].cpu())
        th = torch.full((self.K,), float("inf"))
        for c in range(self.K):
            if per_cls[c]:
                th[c] = torch.quantile(torch.cat(per_cls[c]), self.threshold_pct)
        self.model.thresholds = th.to(self.device)

    def _save(self, epoch, val_result):
        state = {"epoch": epoch, "model_state_dict": self.model.state_dict(),
                 "val_result": val_result, "cfg": self.cfg}
        torch.save(state, os.path.join(self.model_dir, "best_model.pth"))
        logger.info(f"[Ckpt] saved | Score={val_result['score']:.4f} "
                    f"ClosedAcc={val_result['closed_acc']:.4f} "
                    f"AUROC={val_result['open_auroc']}")

    def train(self):
        print(f"\n{'='*60}\n  开始训练 (K={self.K}) | 设备: {self.device}\n{'='*60}")
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            if epoch <= self.warm_up:
                self._warmup_epoch(epoch)
                if epoch == self.warm_up:
                    print("[Trainer] Warm-up 完成, 初始化球心...")
                    self.model.init_centers(self.train_loader, self.device)
                    del self.warmup_head
                continue

            loss = self._train_epoch(epoch)
            if self.sched: self.sched.step()
            self._calibrate()
            v = self._validate()
            auroc = f"{v['open_auroc']:.4f}" if v["open_auroc"] is not None else "N/A"
            print(f"Epoch {epoch:03d}/{self.epochs} | Loss={loss:.4f} | "
                  f"ClosedAcc={v['closed_acc']:.4f} | AUROC={auroc} | "
                  f"Score={v['score']:.4f} | LR={self.opt.param_groups[0]['lr']:.2e} | "
                  f"T={time.time()-t0:.1f}s")

            if v["score"] > self.best_score:
                self.best_score = v["score"]; self.no_improve = 0
                self._save(epoch, v)
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"[早停] {self.patience} 轮无提升, 停止。Best Score={self.best_score:.4f}")
                    break

        print(f"\n{'='*60}\n  训练完成 | Best Score={self.best_score:.4f}\n{'='*60}")


# ═════════════════════════════════════════════
#  主入口
# ═════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    setup_logging(log_file=os.path.join(CFG["output"]["results_dir"], "train.log"))
    set_seed(CFG["data"]["random_seed"])
    device = torch.device(args.device) if args.device else get_device()

    # 数据
    d = CFG["data"]
    train_ds, val_ds = split_train_val(
        d["train_npy"], d["target_classes"], d["unknown_classes"],
        train_ratio=d["train_ratio"], seed=d["random_seed"], normalize=d["normalize"])

    print(f"[Data] Train={len(train_ds)} | Val={len(val_ds)}")
    print(f"[Data] Train 类别分布: {train_ds.get_class_distribution()}")
    print(f"[Data] Val   类别分布: {val_ds.get_class_distribution()}")

    bs = CFG["training"]["batch_size"]; nw = CFG["training"]["num_workers"]
    train_loader = build_dataloader(train_ds, bs, shuffle=True, num_workers=nw)
    val_loader = build_dataloader(val_ds, bs, shuffle=False, num_workers=nw)

    # 模型
    model = MultiSphereDeepSVDDModel(CFG)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] backbone={CFG['model']['backbone']} | "
          f"feature_dim={CFG['model']['feature_dim']} | "
          f"K={model.num_classes} | params={n_params:,}")
    print(f"[Class] targets={d['target_classes']} | knowns_neg={d['unknown_classes']}")

    Trainer(model, train_loader, val_loader, CFG, device).train()
    print("\n训练完成! 运行:  python evaluate.py")


if __name__ == "__main__":
    main()