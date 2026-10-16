"""
evaluate.py
评估入口 - 所有评估相关配置和逻辑都在此文件中

用法:
  python evaluate.py
  python evaluate.py --model models/best_model.pth
"""

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.manifold import TSNE

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import IQDataset, build_dataloader
from src.model import MultiSphereDeepSVDDModel


# ═════════════════════════════════════════════
#  评估配置 (可与 train.py 保持独立)
# ═════════════════════════════════════════════
CFG = {
    "data": {
        "test_npy": "dataset_npy/test_M2048.npy",
        "target_classes":  ["A", "D", "E", "G"],   # 与训练一致
        # 未知类: test 中所有非 target 的类别都视为未知(包括 B/C/F)
        # 显式列出更清晰; 也可设为 None 让 dataset 自动收集
        "unknown_classes": ["B", "C", "F"],
        "normalize": "zscore",
        "random_seed": 42,
    },
    "eval": {
        "batch_size": 128,
        "num_workers": 0,
        "threshold_percentile": 95,   # 阈值分位数
        "val_ratio": 0.3,             # 从 train_npy 里划一份来标定阈值
        "train_npy_for_calib": "dataset_npy/train_M2048.npy",
    },
    "output": {"results_dir": "results"},
}


# ═════════════════════════════════════════════
#  基础工具
# ═════════════════════════════════════════════
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[Device] CPU")
    return torch.device("cpu")


# ═════════════════════════════════════════════
#  评估器
# ═════════════════════════════════════════════
class Evaluator:
    def __init__(self, model, device, cfg):
        self.model = model.eval()
        self.device = device
        self.cfg = cfg
        self.results_dir = cfg["output"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)
        self.target_classes = cfg["data"]["target_classes"]
        self.unknown_classes = cfg["data"]["unknown_classes"] or []
        self.K = model.num_classes

    @torch.no_grad()
    def extract(self, loader):
        feats, dists, labs, clss = [], [], [], []
        for x, y, c in loader:
            x = x.to(self.device)
            f = self.model(x)
            d = self.model.compute_distances(f).sqrt()
            feats.append(f.cpu().numpy()); dists.append(d.cpu().numpy())
            labs.extend(y.numpy()); clss.extend(c)
        feats = np.concatenate(feats); dists = np.concatenate(dists)
        return {
            "feats": feats, "dist_all": dists,
            "min_dist": dists.min(1), "nearest": dists.argmin(1),
            "labels": np.array(labs), "cls_names": clss,
        }

    def calibrate(self, val_res):
        pct = self.cfg["eval"]["threshold_percentile"]
        th = np.full(self.K, np.inf, dtype=np.float32)
        d, y = val_res["dist_all"], val_res["labels"]
        for c in range(self.K):
            m = y == c
            if m.any():
                th[c] = np.percentile(d[m, c], pct)
        self.model.thresholds = torch.tensor(th, device=self.device)
        print("[Evaluator] 阈值校准:")
        for c, name in enumerate(self.target_classes):
            print(f"    [{name}] threshold = {th[c]:.4f}")
        return th

    def _predict(self, res, th):
        cth = th[res["nearest"]]
        is_known = res["min_dist"] < cth
        return np.where(is_known, res["nearest"], -1)

    def compute_metrics(self, res, th):
        y = res["labels"]; md = res["min_dist"]; nc = res["nearest"]
        cls_arr = np.array(res["cls_names"])
        known = y >= 0; unk = y == -1

        closed_acc = float((nc[known] == y[known]).mean()) if known.any() else None

        auroc = fpr95 = None; roc_data = None
        if known.any() and unk.any():
            lab = known.astype(int); sc = -md
            auroc = float(roc_auc_score(lab, sc))
            fpr, tpr, _ = roc_curve(lab, sc)
            fpr95 = float(fpr[np.argmin(np.abs(tpr - 0.95))])
            roc_data = (fpr, tpr)

        preds = self._predict(res, th)

        per_class_recall = {}
        for c, name in enumerate(self.target_classes):
            m = y == c
            per_class_recall[name] = float((preds[m] == c).mean()) if m.any() else None

        per_unknown_reject = {}
        for name in self.unknown_classes:
            m = cls_arr == name
            per_unknown_reject[name] = float((preds[m] == -1).mean()) if m.any() else None
        unk_reject = float((preds[unk] == -1).mean()) if unk.any() else None

        open_acc = float((preds == y).mean())

        cm_true = np.where(y == -1, self.K, y)
        cm_pred = np.where(preds == -1, self.K, preds)
        cm = confusion_matrix(cm_true, cm_pred, labels=list(range(self.K + 1)))

        return {
            "closed_acc": closed_acc, "auroc": auroc, "fpr@95tpr": fpr95,
            "open_set_acc": open_acc, "thresholds": th,
            "per_class_recall": per_class_recall,
            "per_unknown_rejection": per_unknown_reject,
            "unknown_rejection_rate": unk_reject,
            "confusion_matrix": cm,
            "cm_class_names": self.target_classes + ["unknown"],
            "roc_curve": roc_data, "preds": preds,
        }

    def print_report(self, m):
        print("\n" + "="*60 + "\n  多球体开放集识别评估报告\n" + "="*60)
        if m["closed_acc"] is not None:
            print(f"  闭集准确率:                    {m['closed_acc']:.4f}")
        if m["auroc"] is not None:
            print(f"  开集检测 AUROC:                {m['auroc']:.4f}")
            print(f"  FPR@95TPR:                     {m['fpr@95tpr']:.4f}")
        print(f"  开放集整体准确率:              {m['open_set_acc']:.4f}")
        print("-"*60)
        print("  目标类识别率 (应接近 1.0):")
        for n, v in m["per_class_recall"].items():
            if v is not None: print(f"    [{n}] : {v:.4f}")
        print("-"*60)
        if m["unknown_rejection_rate"] is not None:
            print(f"  未知类整体拒识率:              {m['unknown_rejection_rate']:.4f}")
            for n, v in m["per_unknown_rejection"].items():
                if v is not None: print(f"      - [{n}] : {v:.4f}")
        print("-"*60)
        print("  混淆矩阵 (行=真实, 列=预测):")
        names = m["cm_class_names"]
        print("        " + "".join(f"{n[:8]:>10s}" for n in names))
        for i, n in enumerate(names):
            print(f"{n[:8]:>8s}" + "".join(f"{v:>10d}" for v in m["confusion_matrix"][i]))
        print("="*60 + "\n")

    # ─── 绘图: 使用不同 hatch/linestyle 区分同色系类别 ───
    def plot_distance(self, res, th):
        y = res["labels"]; md = res["min_dist"]; d = res["dist_all"]
        cls_arr = np.array(res["cls_names"])

        fig, ax = plt.subplots(figsize=(12, 6))
        # 目标类: 蓝色系, 用不同 hatch 图案区分
        hatches = ["/", "\\", "x", "+", "o", "*", "."]
        base_blue = "steelblue"
        for c, name in enumerate(self.target_classes):
            m = y == c
            if not m.any(): continue
            ax.hist(d[m, c], bins=40, alpha=0.45, density=True,
                    color=base_blue, edgecolor="navy", linewidth=1.2,
                    hatch=hatches[c % len(hatches)],
                    label=f"{name} (目标, 到自身球心)")
            ax.axvline(th[c], color="navy", linestyle="--", linewidth=1.5,
                       alpha=0.6, label=f"{name} 阈值={th[c]:.2f}")

        # 未知类: 红色系, 用不同 hatch 区分
        base_red = "indianred"
        for i, name in enumerate(self.unknown_classes):
            m = cls_arr == name
            if not m.any(): continue
            ax.hist(md[m], bins=40, alpha=0.45, density=True,
                    color=base_red, edgecolor="darkred", linewidth=1.2,
                    hatch=hatches[i % len(hatches)],
                    label=f"{name} (未知, 到最近球心)")

        ax.set_xlabel("到球心的距离")
        ax.set_ylabel("密度")
        ax.set_title("各类信号距离分布 (蓝=目标类, 红=未知类; 图案区分具体类别)")
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        plt.tight_layout()
        p = os.path.join(self.results_dir, "distance_distribution.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[图表] {p}")

    def plot_roc(self, m):
        if m["roc_curve"] is None: return
        fpr, tpr = m["roc_curve"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2, color="steelblue",
                label=f"AUROC={m['auroc']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("开集检测 ROC 曲线 (已知 vs 未知)")
        ax.legend(); plt.tight_layout()
        p = os.path.join(self.results_dir, "roc_curve.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[图表] {p}")

    def plot_tsne(self, res):
        feats = res["feats"]; y = res["labels"]; cls_arr = np.array(res["cls_names"])
        if len(feats) > 2000:
            idx = np.random.choice(len(feats), 2000, replace=False)
            feats, y, cls_arr = feats[idx], y[idx], cls_arr[idx]

        centers = self.model.centers.detach().cpu().numpy()
        emb = TSNE(n_components=2, perplexity=30, max_iter=1000,
                   random_state=42).fit_transform(np.concatenate([feats, centers]))
        emb_pts, emb_ctr = emb[:len(feats)], emb[len(feats):]

        fig, ax = plt.subplots(figsize=(10, 8))
        markers_tgt = ["o", "s", "D", "P", "X", "v", "^"]
        markers_unk = ["*", "+", "x", "1", "2", "3"]

        # 目标类: 蓝色, 不同 marker
        for c, name in enumerate(self.target_classes):
            m = y == c
            if not m.any(): continue
            ax.scatter(emb_pts[m, 0], emb_pts[m, 1], s=25, alpha=0.65,
                       marker=markers_tgt[c % len(markers_tgt)],
                       facecolors="none", edgecolors="steelblue", linewidths=1.2,
                       label=f"{name} (目标)")

        # 未知类: 红色, 不同 marker
        for i, name in enumerate(self.unknown_classes):
            m = cls_arr == name
            if not m.any(): continue
            ax.scatter(emb_pts[m, 0], emb_pts[m, 1], s=30, alpha=0.65,
                       marker=markers_unk[i % len(markers_unk)],
                       color="indianred", label=f"{name} (未知)")

        # 球心: 黑色五角星
        ax.scatter(emb_ctr[:, 0], emb_ctr[:, 1], s=350, marker="*",
                   c="gold", edgecolors="black", linewidths=1.5,
                   label="球心", zorder=10)
        for c, name in enumerate(self.target_classes):
            ax.annotate(name, (emb_ctr[c, 0], emb_ctr[c, 1]),
                        fontsize=10, fontweight="bold", ha="center", va="bottom")

        ax.set_title("t-SNE 特征空间可视化 (蓝=目标类, 红=未知类, ★=球心)")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()
        p = os.path.join(self.results_dir, "tsne_features.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[图表] {p}")


# ═════════════════════════════════════════════
#  主流程
# ═════════════════════════════════════════════
def _to_json(obj):
    if isinstance(obj, dict): return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None: return obj
    return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    setup_logging()
    set_seed(CFG["data"]["random_seed"])
    device = get_device()
    matplotlib.use("Agg")

    # 加载 checkpoint (含训练时的 cfg, 用其重建模型结构)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    state = torch.load(args.model, map_location=device)
    train_cfg = state["cfg"]
    model = MultiSphereDeepSVDDModel(train_cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    print(f"[Model] 已加载: {args.model} (epoch {state['epoch']})")

    d = CFG["data"]; e = CFG["eval"]

    # 校准集: 从训练集里独立划一份 (只保留目标类, 用于阈值分位数)
    from src.dataset import split_train_val
    _, calib_ds = split_train_val(
        e["train_npy_for_calib"], d["target_classes"],
        unknown_classes=[],  # 校准只用目标类
        train_ratio=1 - e["val_ratio"], seed=d["random_seed"],
        normalize=d["normalize"])
    calib_loader = build_dataloader(calib_ds, e["batch_size"], False, e["num_workers"])
    print(f"[Data] 校准集: {len(calib_ds)} | {calib_ds.get_class_distribution()}")

    # 测试集
    test_ds = IQDataset(d["test_npy"], d["target_classes"],
                        unknown_classes=d["unknown_classes"],
                        normalize=d["normalize"])
    test_loader = build_dataloader(test_ds, e["batch_size"], False, e["num_workers"])
    print(f"[Data] 测试集: {len(test_ds)} | {test_ds.get_class_distribution()}")

    ev = Evaluator(model, device, CFG)
    print("\n[Step 1] 阈值校准...")
    val_res = ev.extract(calib_loader)
    th = ev.calibrate(val_res)

    print("\n[Step 2] 测试集评估...")
    test_res = ev.extract(test_loader)
    metrics = ev.compute_metrics(test_res, th)
    ev.print_report(metrics)

    if not args.no_plot:
        ev.plot_distance(test_res, th)
        ev.plot_roc(metrics)
        ev.plot_tsne(test_res)

    save = {k: v for k, v in metrics.items() if k not in ("roc_curve", "preds")}
    out = os.path.join(CFG["output"]["results_dir"], "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_to_json(save), f, indent=2, ensure_ascii=False)
    print(f"[输出] 指标已保存: {out}")


if __name__ == "__main__":
    main()