"""
evaluate.py
评估入口 - 独立单目标超球体集成推理

推理流程:
  1. 加载 K 个独立的 SingleSphereModel (每个对应一个目标类别)
  2. 对每个测试样本, K 个模型各自计算到球心距离
  3. 按距离排序, 最近球体且距离 < 阈值 → 判定为该类; 否则 → unknown
  4. 输出: 排名详情 + 最终判定 + 评估指标 + 图表

  ┌──────────────────────────────────────────────────┐
  │  !! 重要: 以下目标类别必须与 train.py 保持一致 !!  │
  │  target_classes = ["A", "D", "E", "G"]           │
  │  always_negative = ["B"]                          │
  │  test 中额外出现: C, F (训练时从未见过的真未知类)  │
  └──────────────────────────────────────────────────┘

用法:
  python evaluate.py
  python evaluate.py --no-plot
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
from src.model import SingleSphereModel, EnsemblePredictor


# ═════════════════════════════════════════════
#  评估配置
#  !! target_classes 必须与 train.py 中一致 !!
# ═════════════════════════════════════════════
CFG = {
    "data": {
        "test_npy": "data/test_M2048.npy",
        # ── 与 train.py 一致 ──
        "target_classes": ["A", "D", "E", "G"],
        # test 中所有非 target 的类别都视为未知
        # B: 训练时已见过的负类
        # C, F: 训练时从未见过的真未知类
        "unknown_classes": ["B", "C", "F"],
        "normalize": "zscore",
        "random_seed": 42,
    },
    "eval": {
        "batch_size": 128,
        "num_workers": 0,
        # 阈值重新标定 (可选, 用训练集的验证切分)
        "recalibrate": True,
        "threshold_percentile": 95,
        "train_npy_for_calib": "data/train_M2048.npy",
        "always_negative": ["B"],  # 与 train.py 一致
        "val_ratio": 0.3,
    },
    "output": {
        "model_dir": "models",
        "results_dir": "results",
    },
}


# ═════════════════════════════════════════════
#  工具
# ═════════════════════════════════════════════
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_device():
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[Device] CPU")
    return torch.device("cpu")


def _to_json(obj):
    if isinstance(obj, dict): return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_json(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None: return obj
    return str(obj)


# ═════════════════════════════════════════════
#  加载 K 个独立模型
# ═════════════════════════════════════════════
def load_sphere_models(target_classes, model_dir, device):
    """加载所有独立球体模型, 返回 {类名: model} 字典"""
    models = {}
    for name in target_classes:
        path = os.path.join(model_dir, f"sphere_{name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        state = torch.load(path, map_location=device)
        cfg = state["cfg"]
        model = SingleSphereModel(cfg, target_name=name).to(device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        print(f"  [{name}] 已加载 (epoch {state['epoch']}, "
              f"threshold={model.threshold.item():.4f})")
        models[name] = model
    return models


# ═════════════════════════════════════════════
#  阈值重新标定 (可选)
# ═════════════════════════════════════════════
@torch.no_grad()
def recalibrate_thresholds(models, cfg, device):
    """用训练集的验证切分重新标定每个球体的阈值"""
    from src.dataset import split_binary_train_val

    e = cfg["eval"]; d = cfg["data"]
    target_classes = d["target_classes"]
    always_neg = e["always_negative"]
    pct = e["threshold_percentile"] / 100.0

    for name in target_classes:
        model = models[name].eval()
        others = [c for c in target_classes if c != name]
        negs = others + always_neg

        _, val_ds = split_binary_train_val(
            e["train_npy_for_calib"], name, negs,
            train_ratio=1 - e["val_ratio"],
            seed=d["random_seed"], normalize=d["normalize"])

        loader = build_dataloader(val_ds, e["batch_size"])
        dists = []
        for x, y, _ in loader:
            x = x.to(device)
            d_val = model.compute_distance(model(x)).cpu()
            dists.append(d_val[y == 1])  # 只用目标类
        dists = torch.cat(dists)
        th = torch.quantile(dists, pct)
        model.threshold = th.to(device)
        print(f"  [{name}] 重新标定阈值 = {th.item():.4f} "
              f"(p{cfg['eval']['threshold_percentile']})")


# ═════════════════════════════════════════════
#  评估器
# ═════════════════════════════════════════════

class Evaluator:
    def __init__(self, ensemble, cfg):
        self.ensemble = ensemble
        self.cfg = cfg
        self.target_classes = cfg["data"]["target_classes"]
        self.unknown_classes = cfg["data"]["unknown_classes"]
        self.K = len(self.target_classes)
        self.results_dir = cfg["output"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self, test_loader):
        """全量推理 + 计算指标"""
        all_distances, all_pred_label, all_true_label = [], [], []
        all_cls_names, all_details, all_is_known = [], [], []

        for x, y, cls in test_loader:
            result = self.ensemble.predict(x)
            all_distances.append(result["distances"])
            all_pred_label.append(result["pred_label"])
            all_true_label.append(y.numpy())
            all_cls_names.extend(cls)
            all_details.extend(result["details"])
            all_is_known.append(result["is_known"])

        self.distances = np.concatenate(all_distances)     # (N, K)
        self.pred_label = np.concatenate(all_pred_label)   # (N,)
        self.true_label = np.concatenate(all_true_label)   # (N,)
        self.cls_names = np.array(all_cls_names)
        self.is_known = np.concatenate(all_is_known)
        self.min_dist = self.distances.min(axis=1)
        self.details = all_details

        return self._compute_metrics()

    def _compute_metrics(self):
        y = self.true_label; p = self.pred_label; md = self.min_dist
        known = y >= 0; unk = y == -1

        # 闭集: 已知类样本是否被正确分类到对应球体
        # 需要把目标类标签与球体索引对应
        name_to_idx = {n: i for i, n in enumerate(self.target_classes)}
        nearest = self.distances.argmin(axis=1)

        closed_acc = None
        if known.any():
            closed_acc = float((nearest[known] == y[known]).mean())

        # 开集 AUROC
        auroc = fpr95 = None; roc_data = None
        if known.any() and unk.any():
            lab = known.astype(int)
            auroc = float(roc_auc_score(lab, -md))
            fpr, tpr, _ = roc_curve(lab, -md)
            fpr95 = float(fpr[np.argmin(np.abs(tpr - 0.95))])
            roc_data = (fpr, tpr)

        # 逐类识别率
        per_class_recall = {}
        for c, name in enumerate(self.target_classes):
            m = y == c
            per_class_recall[name] = float((p[m] == c).mean()) if m.any() else None

        # 未知类拒识率
        per_unk = {}
        for name in self.unknown_classes:
            m = self.cls_names == name
            per_unk[name] = float((p[m] == -1).mean()) if m.any() else None
        unk_rate = float((p[unk] == -1).mean()) if unk.any() else None

        # 整体准确率
        open_acc = float((p == y).mean())

        # 混淆矩阵
        cm_true = np.where(y == -1, self.K, y)
        cm_pred = np.where(p == -1, self.K, p)
        cm = confusion_matrix(cm_true, cm_pred, labels=list(range(self.K + 1)))
        cm_names = self.target_classes + ["unknown"]

        # 各球体阈值
        thresholds = {n: self.ensemble.models[n].threshold.item()
                      for n in self.target_classes}

        return {
            "closed_acc": closed_acc, "auroc": auroc, "fpr@95tpr": fpr95,
            "open_set_acc": open_acc,
            "per_class_recall": per_class_recall,
            "per_unknown_rejection": per_unk,
            "unknown_rejection_rate": unk_rate,
            "thresholds": thresholds,
            "confusion_matrix": cm, "cm_class_names": cm_names,
            "roc_curve": roc_data,
        }

    def print_report(self, m):
        print(f"\n{'='*60}\n  独立超球体 开放集识别评估报告\n{'='*60}")
        if m["closed_acc"] is not None:
            print(f"  闭集准确率:              {m['closed_acc']:.4f}")
        if m["auroc"] is not None:
            print(f"  开集 AUROC:              {m['auroc']:.4f}")
            print(f"  FPR@95TPR:               {m['fpr@95tpr']:.4f}")
        print(f"  开放集整体准确率:        {m['open_set_acc']:.4f}")
        print("-"*60)
        print("  目标类识别率:")
        for n, v in m["per_class_recall"].items():
            if v is not None: print(f"    [{n}] : {v:.4f}")
        print("-"*60)
        if m["unknown_rejection_rate"] is not None:
            print(f"  未知类整体拒识率:        {m['unknown_rejection_rate']:.4f}")
            for n, v in m["per_unknown_rejection"].items():
                if v is not None: print(f"      [{n}] : {v:.4f}")
        print("-"*60)
        print("  各球体阈值:")
        for n, th in m["thresholds"].items():
            print(f"    [{n}] threshold = {th:.4f}")
        print("-"*60)
        print("  混淆矩阵 (行=真实, 列=预测):")
        names = m["cm_class_names"]
        print("        " + "".join(f"{n:>10s}" for n in names))
        for i, n in enumerate(names):
            print(f"{n:>8s}" + "".join(f"{v:>10d}" for v in m["confusion_matrix"][i]))
        print("="*60)

    def print_sample_details(self, n=5):
        """打印前 n 个样本的推理排名详情"""
        print(f"\n{'─'*60}")
        print(f"  前 {n} 个样本推理详情:")
        print(f"{'─'*60}")
        for i in range(min(n, len(self.details))):
            true_cls = self.cls_names[i]
            true_lab = self.true_label[i]
            pred_lab = self.pred_label[i]
            pred_cls = self.target_classes[pred_lab] if pred_lab >= 0 else "unknown"
            correct = "✓" if pred_lab == true_lab else "✗"

            print(f"\n  样本 {i+1} | 真实={true_cls}(label={true_lab}) | "
                  f"预测={pred_cls}(label={pred_lab}) {correct}")
            print(f"  {'─'*50}")
            detail = self.details[i]
            for item in detail[:-1]:  # 排名信息
                marker = "●" if item["within_sphere"] else "○"
                print(f"    排名{item['rank']}: [{item['class']}] "
                      f"距离={item['distance']:.4f} "
                      f"阈值={item['threshold']:.4f} "
                      f"{marker}")
            unk_info = detail[-1]
            print(f"    未知度(最近距离/阈值): {unk_info['unknown_ratio']:.4f} "
                  f"{'→ 判定未知' if unk_info['is_unknown'] else '→ 判定已知'}")

    # ─── 绘图 ───
    def plot_distance(self, metrics):
        fig, ax = plt.subplots(figsize=(12, 6))
        # 每个目标类用独立颜色 + hatch
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
                  "#F44336", "#00BCD4", "#795548"]
        hatches = ["/", "\\", "x", "+", "o", "*"]

        for c, name in enumerate(self.target_classes):
            m = self.true_label == c
            if not m.any(): continue
            ax.hist(self.distances[m, c], bins=40, alpha=0.5, density=True,
                    color=colors[c % len(colors)],
                    edgecolor="black", linewidth=0.8,
                    hatch=hatches[c % len(hatches)],
                    label=f"{name} (目标→球{name})")
            th = metrics["thresholds"][name]
            ax.axvline(th, color=colors[c % len(colors)], linestyle="--",
                       linewidth=2, label=f"{name} 阈值={th:.2f}")

        # 未知类: 到最近球心距离
        unk_colors = ["#E91E63", "#607D8B", "#FF5722"]
        unk_hatches = [".", "O", "//"]
        for i, name in enumerate(self.unknown_classes):
            m = self.cls_names == name
            if not m.any(): continue
            ax.hist(self.min_dist[m], bins=40, alpha=0.4, density=True,
                    color=unk_colors[i % len(unk_colors)],
                    edgecolor="black", linewidth=0.8,
                    hatch=unk_hatches[i % len(unk_hatches)],
                    label=f"{name} (未知→最近球)")

        ax.set_xlabel("到球心的距离", fontsize=12)
        ax.set_ylabel("密度", fontsize=12)
        ax.set_title("各类信号距离分布 (每种颜色+图案=一个类别; 虚线=阈值)")
        ax.legend(fontsize=8, ncol=2); plt.tight_layout()
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
        ax.set_title("ROC 曲线 (已知 vs 未知)")
        ax.legend(); plt.tight_layout()
        p = os.path.join(self.results_dir, "roc_curve.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[图表] {p}")

    def plot_tsne(self):
        """
        t-SNE: 用所有 K 个模型分别提取特征, 取第一个模型的特征空间
        (各模型特征空间不同, 此处仅做可视化参考)
        """
        # 取第一个模型的特征做可视化
        ref_name = self.target_classes[0]
        ref_model = self.ensemble.models[ref_name]

        # 采样
        idx = np.arange(len(self.cls_names))
        if len(idx) > 2000:
            idx = np.random.choice(idx, 2000, replace=False)

        # 直接用已有距离矩阵做可视化 (距离矩阵是跨模型的, 更有意义)
        # 每个样本有 K 维距离向量, 可直接用 t-SNE 降维
        dist_subset = self.distances[idx]
        y_sub = self.true_label[idx]
        cls_sub = self.cls_names[idx]

        emb = TSNE(n_components=2, perplexity=30, max_iter=1000,
                   random_state=42).fit_transform(dist_subset)

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        markers_t = ["o", "s", "D", "P"]
        unk_colors = ["#E91E63", "#607D8B", "#FF5722"]
        markers_u = ["^", "v", "<"]

        fig, ax = plt.subplots(figsize=(10, 8))
        for c, name in enumerate(self.target_classes):
            m = y_sub == c
            if not m.any(): continue
            ax.scatter(emb[m, 0], emb[m, 1], s=25, alpha=0.65,
                       c=colors[c % len(colors)],
                       marker=markers_t[c % len(markers_t)],
                       label=f"{name} (目标)")

        for i, name in enumerate(self.unknown_classes):
            m = cls_sub == name
            if not m.any(): continue
            ax.scatter(emb[m, 0], emb[m, 1], s=30, alpha=0.55,
                       c=unk_colors[i % len(unk_colors)],
                       marker=markers_u[i % len(markers_u)],
                       label=f"{name} (未知)")

        ax.set_title("t-SNE (基于各球体距离向量)\n"
                     "每个样本用 [d(球A), d(球D), d(球E), d(球G)] 降维")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()
        p = os.path.join(self.results_dir, "tsne_distances.png")
        plt.savefig(p, dpi=150); plt.close()
        print(f"[图表] {p}")


# ═════════════════════════════════════════════
#  主流程
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show-samples", type=int, default=10,
                        help="打印前 N 个样本的推理详情")
    args = parser.parse_args()

    setup_logging()
    set_seed(CFG["data"]["random_seed"])
    device = get_device()
    matplotlib.use("Agg")

    d = CFG["data"]; e = CFG["eval"]
    target_classes = d["target_classes"]

    print(f"\n{'='*60}")
    print(f"  独立超球体 集成评估")
    print(f"  目标类: {target_classes}")
    print(f"  未知类: {d['unknown_classes']}")
    print(f"{'='*60}")

    # 加载 K 个模型
    print("\n[Step 1] 加载模型...")
    models = load_sphere_models(target_classes, CFG["output"]["model_dir"], device)

    # 阈值重新标定 (可选)
    if e["recalibrate"]:
        print("\n[Step 2] 重新标定阈值...")
        recalibrate_thresholds(models, CFG, device)

    # 构建集成推理器
    ensemble = EnsemblePredictor(models, device)

    # 测试集
    print("\n[Step 3] 加载测试集...")
    test_ds = IQDataset(d["test_npy"], target_classes,
                        unknown_classes=d["unknown_classes"],
                        normalize=d["normalize"])
    test_loader = build_dataloader(test_ds, e["batch_size"], num_workers=e["num_workers"])
    print(f"  测试集: {len(test_ds)} | {test_ds.get_class_distribution()}")

    # 推理
    print("\n[Step 4] 集成推理...")
    evaluator = Evaluator(ensemble, CFG)
    metrics = evaluator.run(test_loader)

    # 报告
    evaluator.print_report(metrics)
    if args.show_samples > 0:
        evaluator.print_sample_details(args.show_samples)

    # 绘图
    if not args.no_plot:
        evaluator.plot_distance(metrics)
        evaluator.plot_roc(metrics)
        evaluator.plot_tsne()

    # 保存指标
    save = {k: v for k, v in metrics.items() if k != "roc_curve"}
    out = os.path.join(CFG["output"]["results_dir"], "metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_to_json(save), f, indent=2, ensure_ascii=False)
    print(f"\n[输出] 指标: {out}")


if __name__ == "__main__":
    main()