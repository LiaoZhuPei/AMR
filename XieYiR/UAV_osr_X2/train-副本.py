"""
train.py
训练入口 - 独立单目标超球体模式

核心思想:
  对每个目标类别 (如 A, D, E, G) 分别训练一个独立的 SingleSphereModel:
    - 以该类为目标 (label=1)
    - 以其余所有类为负类 (label=0), 包括:
        其他目标类 (如训练 A 时, D/E/G 为负类)
        始终作为负类的类别 (如 B)
    - 权重每次完全重置, 不共享

  训练完成后保存 K 个独立模型, evaluate.py 分别加载做集成推理。

用法:
  python train.py
  python train.py --device cuda
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import split_binary_train_val, build_dataloader
from src.model import SingleSphereModel, SingleSphereSVDDLoss

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════
#  配置
# ═════════════════════════════════════════════
CFG = {
    "data": {
        "train_npy": "data/train_M2048.npy",
        #  目标类别: 每个类别将独立训练一个超球体
        "target_classes": ["A", "D", "E", "G"],
        #  始终作为负类的类别 (所有球体训练时都视为负类)
        "always_negative": ["BACKGROUND"],
        "train_ratio": 0.8,
        "random_seed": 42,
        "normalize": "zscore",
    },
    "model": {
        "backbone": "cnn1d",
        "feature_dim": 128,
        "normalize_feature": True,
        "cnn1d": {
            "channels": [32, 64, 128],
            "kernel_sizes": [7, 5, 3],
            "pool_sizes": [2, 2, 2],
        },
        "cnn2d": {"channels": [16, 32, 64, 128]},
    },
    "training": {
        "epochs": 80,
        "batch_size": 128,
        "num_workers": 0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "patience": 10,
        "lr_scheduler": "cosine",
        "svdd": {
            "warm_up_epochs": 10,
            "margin": 4.0,
            "lambda_neg": 1.0,
            "update_center_every": 5,
            "center_momentum": 0.9,
            "threshold_percentile": 0.95,
        },
    },
    "output": {
        "model_dir": "models",
        "results_dir": "results",
    },
}


# ═════════════════════════════════════════════
#  工具
# ═════════════════════════════════════════════
def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[Device] CPU")
    return torch.device("cpu")


# ═════════════════════════════════════════════
#  单球体训练器
# ═════════════════════════════════════════════

class SingleSphereTrainer:
    """
    为单个目标类别训练一个 SingleSphereModel。

    训练流程:
      1. Warm-up: 二分类预训练 (目标=1, 负=0), 让网络学到基本的分离能力
      2. 球心初始化: 用目标类样本特征均值初始化
      3. SVDD 训练: 目标聚拢 + 负类排斥
      4. 阈值标定: 用验证集中目标类样本的距离分位数设定阈值
    """

    def __init__(self, model, train_loader, val_loader, target_name, cfg, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_name = target_name
        self.cfg = cfg
        self.device = device

        tc = cfg["training"]; sc = tc["svdd"]
        self.epochs = tc["epochs"]
        self.warm_up = sc["warm_up_epochs"]
        self.patience = tc["patience"]
        self.margin = sc["margin"]
        self.update_every = sc["update_center_every"]
        self.center_mom = sc["center_momentum"]
        self.threshold_pct = sc["threshold_percentile"]

        self.svdd_loss = SingleSphereSVDDLoss(
            lambda_neg=sc["lambda_neg"], margin=sc["margin"])

        # Warm-up 分类头
        self.warmup_head = nn.Linear(model.feature_dim, 2).to(device)
        self.warmup_opt = optim.Adam(
            list(model.parameters()) + list(self.warmup_head.parameters()),
            lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        self.opt = optim.Adam(
            model.parameters(),
            lr=tc["learning_rate"], weight_decay=tc["weight_decay"])

        svdd_epochs = max(self.epochs - self.warm_up, 1)
        if tc["lr_scheduler"] == "cosine":
            self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=svdd_epochs)
        else:
            self.sched = None

        self.best_score = -1.0
        self.no_improve = 0
        self.model_dir = cfg["output"]["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    def _warmup_epoch(self, epoch):
        self.model.train(); self.warmup_head.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y, _ in self.train_loader:
            x, y = x.to(self.device), y.to(self.device).long()
            logits = self.warmup_head(self.model(x))
            loss = F.cross_entropy(logits, y)
            self.warmup_opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.warmup_head.parameters()), 1.0)
            self.warmup_opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        logger.info(f"  [{self.target_name}] Warm-up {epoch:03d} | "
                    f"Loss={total_loss/len(self.train_loader):.4f} | "
                    f"Acc={correct/max(n,1):.4f}")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        target_feats = []
        for x, y, _ in self.train_loader:
            x, y = x.to(self.device), y.to(self.device).long()
            feat = self.model(x)
            dist_sq = self.model.compute_distance_sq(feat)
            loss = self.svdd_loss(dist_sq, y)

            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total_loss += loss.item()

            if self.update_every > 0:
                with torch.no_grad():
                    m = y == 1
                    if m.any():
                        target_feats.append(feat[m].detach().cpu())

        if self.update_every > 0 and epoch % self.update_every == 0 and target_feats:
            self._ema_update_center(target_feats)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _ema_update_center(self, feats):
        new_mean = torch.cat(feats, 0).mean(0)
        if self.model.normalize_feat:
            new_mean = F.normalize(new_mean, dim=0, eps=1e-8)
        old = self.model.center.cpu()
        merged = self.center_mom * old + (1 - self.center_mom) * new_mean
        if self.model.normalize_feat:
            merged = F.normalize(merged, dim=0, eps=1e-8)
        self.model.center = merged.to(self.device)

    @torch.no_grad()
    def _validate(self):
        """返回验证指标: 目标类到球心平均距离, 负类到球心平均距离, 分离度"""
        self.model.eval()
        tgt_d, neg_d = [], []
        for x, y, _ in self.val_loader:
            x = x.to(self.device)
            d = self.model.compute_distance(self.model(x)).cpu()
            tgt_d.append(d[y == 1]); neg_d.append(d[y == 0])
        tgt_d = torch.cat(tgt_d) if any(len(t) > 0 for t in tgt_d) else torch.tensor([0.0])
        neg_d = torch.cat(neg_d) if any(len(t) > 0 for t in neg_d) else torch.tensor([0.0])

        mean_tgt = tgt_d.mean().item()
        mean_neg = neg_d.mean().item()
        # 分离度: 负类平均距离 / 目标类平均距离, 越大越好
        separation = mean_neg / (mean_tgt + 1e-8)
        return {"mean_tgt_dist": mean_tgt, "mean_neg_dist": mean_neg,
                "separation": separation}

    @torch.no_grad()
    def _calibrate_threshold(self):
        """用验证集中目标类样本到球心距离的分位数标定阈值"""
        self.model.eval()
        dists = []
        for x, y, _ in self.val_loader:
            x = x.to(self.device)
            d = self.model.compute_distance(self.model(x)).cpu()
            dists.append(d[y == 1])
        dists = torch.cat(dists)
        th = torch.quantile(dists, self.threshold_pct)
        self.model.threshold = th.to(self.device)
        return th.item()

    def _save(self, epoch, val_result):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "target_name": self.target_name,
            "val_result": val_result,
            "cfg": self.cfg,
        }
        path = os.path.join(self.model_dir, f"sphere_{self.target_name}.pth")
        torch.save(state, path)
        logger.info(f"  [{self.target_name}] 模型已保存 → {path}")

    def train(self):
        print(f"\n  ┌─ 开始训练球体: {self.target_name} ─┐")
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            if epoch <= self.warm_up:
                self._warmup_epoch(epoch)
                if epoch == self.warm_up:
                    print(f"  [{self.target_name}] Warm-up 完成, 初始化球心...")
                    self.model.init_center(self.train_loader, self.device)
                    del self.warmup_head
                continue

            loss = self._train_epoch(epoch)
            if self.sched:
                self.sched.step()

            th = self._calibrate_threshold()
            v = self._validate()

            score = v["separation"]
            print(f"  [{self.target_name}] Epoch {epoch:03d}/{self.epochs} | "
                  f"Loss={loss:.4f} | TgtDist={v['mean_tgt_dist']:.4f} | "
                  f"NegDist={v['mean_neg_dist']:.4f} | Sep={score:.2f} | "
                  f"Th={th:.4f} | T={time.time()-t0:.1f}s")

            if score > self.best_score:
                self.best_score = score
                self.no_improve = 0
                self._save(epoch, v)
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"  [{self.target_name}] 早停, Best Sep={self.best_score:.2f}")
                    break

        print(f"  └─ {self.target_name} 训练完成, Best Sep={self.best_score:.2f} ─┘")


# ═════════════════════════════════════════════
#  主入口: 依次训练 K 个独立球体
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    setup_logging(os.path.join(CFG["output"]["results_dir"], "train.log"))
    set_seed(CFG["data"]["random_seed"])
    device = torch.device(args.device) if args.device else get_device()

    d = CFG["data"]
    target_classes = d["target_classes"]
    always_neg = d["always_negative"]

    print(f"\n{'='*60}")
    print(f"  独立单目标超球体训练")
    print(f"  目标类别: {target_classes}")
    print(f"  始终负类: {always_neg}")
    print(f"  将依次训练 {len(target_classes)} 个独立模型")
    print(f"{'='*60}")

    for target in target_classes:
        # 对于目标类 target, 负类 = 其他目标类 + 始终负类
        other_targets = [c for c in target_classes if c != target]
        negatives = other_targets + always_neg

        print(f"\n{'─'*60}")
        print(f"  目标: {target} | 负类: {negatives}")
        print(f"{'─'*60}")

        # 数据切分
        train_ds, val_ds = split_binary_train_val(
            d["train_npy"], target, negatives,
            train_ratio=d["train_ratio"],
            seed=d["random_seed"],
            normalize=d["normalize"])

        print(f"  Train: {len(train_ds)} | {train_ds.get_class_distribution()}")
        print(f"  Val:   {len(val_ds)}   | {val_ds.get_class_distribution()}")

        bs = CFG["training"]["batch_size"]
        nw = CFG["training"]["num_workers"]
        train_loader = build_dataloader(train_ds, bs, shuffle=True, num_workers=nw)
        val_loader = build_dataloader(val_ds, bs, shuffle=False, num_workers=nw)

        # 每个目标类别创建全新模型 (权重完全独立)
        set_seed(CFG["data"]["random_seed"])  # 重置种子保证可复现
        model = SingleSphereModel(CFG, target_name=target)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  模型参数量: {n_params:,}")

        trainer = SingleSphereTrainer(
            model, train_loader, val_loader, target, CFG, device)
        trainer.train()

    print(f"\n{'='*60}")
    print(f"  全部 {len(target_classes)} 个球体训练完成!")
    print(f"  模型保存在: {CFG['output']['model_dir']}/")
    print(f"  运行:  python evaluate.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()