"""
evaluator.py
评估模块（多球体 Deep SVDD 版本）

功能:
  1. 提取全量测试集特征 + 到各球心的距离矩阵
  2. 阈值校准 (按已知类别分别校准，基于验证集分位数约束)
  3. 计算开放集评估指标:
     - 闭集多分类准确率 (已知类内部是否分类正确)
     - 开集检测 AUROC / FPR@95TPR (已知 vs 未知)
     - 逐类识别率 / 未知类拒识率
     - 混淆矩阵 (含 "unknown" 列)
  4. 绘图:
     - 距离分布直方图 (各已知类 + 未知类)
     - ROC 曲线 (已知 vs 未知)
     - t-SNE 特征可视化 (含球心位置)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    多球体开放集识别评估器。

    Args:
        model:    MultiSphereDeepSVDDModel (已加载权重)
        device:   torch.device
        cfg:      完整配置字典

    约定:
    cfg["data"]["target_classes"]: list[str]，长度为 K，
        下标即对应训练时的整数标签 0..K-1（即模型球心对应的类别）
    cfg["data"]["known_negative_classes"]: list[str]，
        训练/验证/测试中均可能出现的已知非目标类别，
        这些样本的整数标签统一为 -1
    cfg["data"]["unknown_classes"]: list[str]，
        仅出现在测试集(test_unknown.npy)中的全新未知类别，
        训练时从未见过，这些样本的整数标签同样统一为 -1
    """

    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.results_dir = cfg["output"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

        # 修正: 使用 config.yaml 中实际存在的字段名 target_classes
        # 这是唯一拥有真实球心/标签(0..K-1)的类别列表
        self.known_classes = cfg["data"]["target_classes"]

        # known_negative_classes 和 unknown_classes 在标签体系中都统一为 -1，
        # Evaluator 内部不需要区分二者的来源，仅在绘图时可选地做展示区分
        self.known_negative_classes = cfg["data"].get("known_negative_classes", [])
        self.unknown_classes = cfg["data"].get("unknown_classes", [])

        self.num_classes = model.num_classes

    # ─────────────────────────────────────────────
    #  特征提取
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, data_loader) -> dict:
        """
        从 DataLoader 中提取所有样本的特征和到各球心的距离。

        Returns:
            dict with keys:
              "feats"         : np.ndarray (N, D)
              "dist_all"      : np.ndarray (N, K) 到每个球心的距离
              "min_dist"      : np.ndarray (N,)   到最近球心的距离
              "nearest_class" : np.ndarray (N,)   最近球心的类别索引
              "labels"        : np.ndarray (N,)   真实标签，-1表示未知类
              "cls_names"     : list[str]         每个样本原始类名(用于绘图分组)
        """
        self.model.eval()
        all_feats, all_dist_all, all_labels, all_cls = [], [], [], []

        for x, labels, cls_names in data_loader:
            x = x.to(self.device)
            feat = self.model(x)
            dist_sq_all = self.model.compute_distances(feat)  # (B, K)
            dist_all = dist_sq_all.sqrt()

            all_feats.append(feat.cpu().numpy())
            all_dist_all.append(dist_all.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cls.extend(cls_names)

        feats = np.concatenate(all_feats, axis=0)
        dist_all = np.concatenate(all_dist_all, axis=0)
        labels = np.array(all_labels)

        min_dist = dist_all.min(axis=1)
        nearest_class = dist_all.argmin(axis=1)

        return {
            "feats": feats,
            "dist_all": dist_all,
            "min_dist": min_dist,
            "nearest_class": nearest_class,
            "labels": labels,
            "cls_names": all_cls
        }

    # ─────────────────────────────────────────────
    #  阈值校准（按类别）
    # ─────────────────────────────────────────────

    def calibrate_threshold(self, val_results: dict) -> np.ndarray:
        """
        在验证集上按已知类别分别校准拒识阈值。

        策略: 对每个已知类别 c，取该类样本"到自身球心"的距离分布，
              用指定分位数 (默认95%) 作为该类阈值，
              即允许约 5% 的该类正常样本被误判为未知，换取更严格的拒识边界。

        Args:
            val_results: extract_features() 的输出，
                         其中 labels 应为已知类的验证集（不含 -1）

        Returns:
            thresholds: np.ndarray (K,)
        """
        eval_cfg = self.cfg["evaluation"]
        strategy = eval_cfg["threshold_strategy"]
        percentile = eval_cfg.get("percentile", 95)

        dist_all = val_results["dist_all"]     # (N, K)
        labels = val_results["labels"]         # (N,)

        thresholds = np.full(self.num_classes, np.inf, dtype=np.float32)

        for c in range(self.num_classes):
            mask_c = (labels == c)
            if mask_c.sum() == 0:
                logger.warning(f"[Evaluator] 类别 {c} 在验证集中没有样本，"
                                f"阈值保持为 inf")
                continue
            # 该类样本到"自身类别球心"的距离
            own_dist = dist_all[mask_c, c]

            if strategy in ("val_fpr", "percentile"):
                thresholds[c] = np.percentile(own_dist, percentile)
            else:
                raise ValueError(f"Unknown threshold strategy: {strategy}")

        print(f"[Evaluator] 阈值校准完成 (策略: {strategy}, "
              f"分位数: {percentile}):")
        for c, name in enumerate(self.known_classes):
            print(f"    [{name}] threshold = {thresholds[c]:.4f}")

        # 写入模型 buffer
        self.model.thresholds = torch.tensor(
            thresholds, device=self.device, dtype=torch.float32
        )
        return thresholds

    # ─────────────────────────────────────────────
    #  决策：结合最近球心 + 逐类阈值 得到最终预测
    # ─────────────────────────────────────────────

    def _make_predictions(self, dist_all: np.ndarray,
                          nearest_class: np.ndarray,
                          min_dist: np.ndarray,
                          thresholds: np.ndarray) -> np.ndarray:
        """
        Args:
            dist_all:      (N, K)
            nearest_class: (N,)
            min_dist:      (N,)
            thresholds:    (K,)
        Returns:
            preds: (N,) 预测类别，-1 表示判定为未知
        """
        class_threshold = thresholds[nearest_class]
        is_known = min_dist < class_threshold
        preds = np.where(is_known, nearest_class, -1)
        return preds

    # ─────────────────────────────────────────────
    #  评估指标（增强版：区分已知负类 vs 真未知类）
    # ─────────────────────────────────────────────

    def compute_metrics(self, test_results: dict,
                        thresholds: np.ndarray) -> dict:
        """
        计算全套开放集评估指标。

        相比基础版新增:
        - known_neg_rejection_rate: 已知负类(训练时可见)的拒识率
        - unknown_rejection_rate:   真未知类(训练时从未见过)的拒识率
        两者分开统计，因为难度通常不同，合并统计会掩盖问题

        Args:
            test_results: extract_features() 输出
            thresholds:   (K,) 逐类阈值
        """
        dist_all = test_results["dist_all"]
        min_dist = test_results["min_dist"]
        nearest_class = test_results["nearest_class"]
        labels = test_results["labels"]
        cls_names_arr = np.array(test_results["cls_names"])

        known_mask = labels >= 0
        unknown_mask = labels == -1   # 所有 label=-1 的样本(已知负类+真未知类)

        # ── 细分 label=-1 样本的来源 ──
        is_known_neg = np.array([c in self.known_negative_classes for c in cls_names_arr])
        is_true_unknown = np.array([c in self.unknown_classes for c in cls_names_arr])

        # ── 闭集多分类准确率 ──
        if known_mask.sum() > 0:
            closed_acc = float(
                (nearest_class[known_mask] == labels[known_mask]).mean()
            )
        else:
            closed_acc = None

        # ── 开集检测 AUROC / FPR@95TPR（已知 vs 全部-1样本）──
        auroc = None
        fpr_at_95tpr = None
        roc_data = None
        if known_mask.sum() > 0 and unknown_mask.sum() > 0:
            binary_labels = known_mask.astype(np.int32)
            scores = -min_dist
            auroc = roc_auc_score(binary_labels, scores)

            fpr_arr, tpr_arr, _ = roc_curve(binary_labels, scores)
            idx = np.argmin(np.abs(tpr_arr - 0.95))
            fpr_at_95tpr = float(fpr_arr[idx])
            roc_data = (fpr_arr, tpr_arr)

        # ── 最终决策 ──
        preds = self._make_predictions(dist_all, nearest_class, min_dist, thresholds)

        # ── 逐类识别率（已知目标类）──
        per_class_recall = {}
        for c, name in enumerate(self.known_classes):
            mask_c = (labels == c)
            if mask_c.sum() == 0:
                per_class_recall[name] = None
                continue
            correct = (preds[mask_c] == c).sum()
            per_class_recall[name] = float(correct / mask_c.sum())

        # ── 已知负类拒识率（分开统计，且逐类展开）──
        known_neg_rejection_rate = None
        per_known_neg_rejection = {}
        if is_known_neg.sum() > 0:
            known_neg_rejection_rate = float((preds[is_known_neg] == -1).mean())
            for name in self.known_negative_classes:
                mask_c = (cls_names_arr == name)
                if mask_c.sum() == 0:
                    per_known_neg_rejection[name] = None
                    continue
                per_known_neg_rejection[name] = float((preds[mask_c] == -1).mean())

        # ── 真未知类拒识率（分开统计，且逐类展开）──
        unknown_rejection_rate = None
        per_unknown_rejection = {}
        if is_true_unknown.sum() > 0:
            unknown_rejection_rate = float((preds[is_true_unknown] == -1).mean())
            for name in self.unknown_classes:
                mask_c = (cls_names_arr == name)
                if mask_c.sum() == 0:
                    per_unknown_rejection[name] = None
                    continue
                per_unknown_rejection[name] = float((preds[mask_c] == -1).mean())

        # ── 开放集整体准确率 ──
        open_set_acc = float((preds == labels).mean())

        # ── 混淆矩阵（已知类 0..K-1 + "unknown"作为第K类）──
        cm_labels_true = np.where(labels == -1, self.num_classes, labels)
        cm_labels_pred = np.where(preds == -1, self.num_classes, preds)
        cm_class_names = self.known_classes + ["unknown"]
        cm = confusion_matrix(
            cm_labels_true, cm_labels_pred,
            labels=list(range(self.num_classes + 1))
        )

        metrics = {
            "closed_acc": closed_acc,
            "auroc": auroc,
            "fpr_at_95tpr": fpr_at_95tpr,
            "thresholds": thresholds,
            "per_class_recall": per_class_recall,
            "known_neg_rejection_rate": known_neg_rejection_rate,
            "per_known_neg_rejection": per_known_neg_rejection,
            "unknown_rejection_rate": unknown_rejection_rate,
            "per_unknown_rejection": per_unknown_rejection,
            "open_set_acc": open_set_acc,
            "confusion_matrix": cm,
            "cm_class_names": cm_class_names,
            "roc_curve": roc_data,
            "preds": preds
        }

        return metrics


    # ─────────────────────────────────────────────
    #  打印报告（增强版）
    # ─────────────────────────────────────────────

    def print_report(self, metrics: dict):
        print("\n" + "="*60)
        print("  多球体开放集识别评估报告")
        print("="*60)

        if metrics["closed_acc"] is not None:
            print(f"  闭集多分类准确率 (Closed-set Acc): "
                f"{metrics['closed_acc']:.4f}")

        if metrics["auroc"] is not None:
            print(f"  开集检测 AUROC (已知 vs 全部非目标)  : "
                f"{metrics['auroc']:.4f}")
            print(f"  FPR@95TPR                          : "
                f"{metrics['fpr_at_95tpr']:.4f}")

        print(f"  开放集整体准确率 (含拒识)            : "
            f"{metrics['open_set_acc']:.4f}")
        print("-"*60)

        print("  逐类识别率 (已知目标类，应接近 1.0):")
        for name, recall in metrics["per_class_recall"].items():
            if recall is not None:
                print(f"    [{name}] : {recall:.4f}")

        print("-"*60)
        if metrics["known_neg_rejection_rate"] is not None:
            print(f"  [已知负类] 整体拒识率              : "
                f"{metrics['known_neg_rejection_rate']:.4f}  "
                f"← 应接近 1.0（训练时已见过，理论上更容易拒识）")
            for name, rate in metrics["per_known_neg_rejection"].items():
                if rate is not None:
                    print(f"      - [{name}] : {rate:.4f}")

        print("-"*60)
        if metrics["unknown_rejection_rate"] is not None:
            print(f"  [真未知类] 整体拒识率 (关键指标)    : "
                f"{metrics['unknown_rejection_rate']:.4f}  "
                f"← 应接近 1.0（训练时从未见过，更具挑战性）")
            for name, rate in metrics["per_unknown_rejection"].items():
                if rate is not None:
                    print(f"      - [{name}] : {rate:.4f}")

        print("-"*60)
        print("  逐类阈值:")
        for name, th in zip(self.known_classes, metrics["thresholds"]):
            print(f"    [{name}] threshold = {th:.4f}")

        print("-"*60)
        print("  混淆矩阵 (行=真实, 列=预测, 最后一行/列=unknown):")
        cm = metrics["confusion_matrix"]
        names = metrics["cm_class_names"]
        header = "        " + "".join(f"{n[:8]:>10s}" for n in names)
        print(header)
        for i, name in enumerate(names):
            row = "".join(f"{v:>10d}" for v in cm[i])
            print(f"{name[:8]:>8s}{row}")

        print("="*60 + "\n")


    # ─────────────────────────────────────────────
    #  可视化（增强版：区分已知负类 vs 真未知类）
    # ─────────────────────────────────────────────

    def plot_distance_distribution(self, test_results: dict,
                                thresholds: np.ndarray,
                                save: bool = True):
        """
        绘制各类信号的距离分布直方图。
        - 已知目标类: 使用"到自身类别球心"的距离
        - 已知负类:   使用"到最近球心"的距离，暖色系区分具体类别
        - 真未知类:   使用"到最近球心"的距离，冷色系区分具体类别
        """
        dist_all = test_results["dist_all"]
        min_dist = test_results["min_dist"]
        labels = test_results["labels"]
        cls_names_arr = np.array(test_results["cls_names"])

        fig, ax = plt.subplots(figsize=(12, 6))

        # ── 已知目标类：到自身球心的距离 ──
        colors_known = plt.cm.Blues(np.linspace(0.4, 0.9, max(self.num_classes, 1)))
        for c, name in enumerate(self.known_classes):
            mask_c = (labels == c)
            if mask_c.sum() == 0:
                continue
            own_dist = dist_all[mask_c, c]
            ax.hist(own_dist, bins=50, alpha=0.6,
                    label=f"{name} (目标类)",
                    color=colors_known[c], density=True)
            ax.axvline(thresholds[c], color=colors_known[c],
                    linestyle="--", linewidth=1.5,
                    label=f"{name} 阈值={thresholds[c]:.2f}")

        # ── 已知负类：到最近球心的距离，暖色系 ──
        if len(self.known_negative_classes) > 0:
            colors_kn = plt.cm.Oranges(np.linspace(0.4, 0.9, len(self.known_negative_classes)))
            for i, name in enumerate(self.known_negative_classes):
                mask_c = (cls_names_arr == name)
                if mask_c.sum() == 0:
                    continue
                ax.hist(min_dist[mask_c], bins=50, alpha=0.5,
                        label=f"{name} (已知负类)",
                        color=colors_kn[i], density=True)

        # ── 真未知类：到最近球心的距离，冷色系(绿/紫) ──
        if len(self.unknown_classes) > 0:
            colors_unk = plt.cm.Greens(np.linspace(0.4, 0.9, len(self.unknown_classes)))
            for i, name in enumerate(self.unknown_classes):
                mask_c = (cls_names_arr == name)
                if mask_c.sum() == 0:
                    continue
                ax.hist(min_dist[mask_c], bins=50, alpha=0.5,
                        label=f"{name} (真未知类)",
                        color=colors_unk[i], density=True)

        ax.set_xlabel("到球心的距离", fontsize=12)
        ax.set_ylabel("密度", fontsize=12)
        ax.set_title("各类信号距离分布\n"
                    "(蓝色=已知目标类到自身球心, 橙色=已知负类, 绿色=真未知类, "
                    "均为到最近球心距离；虚线=各目标类阈值)",
                    fontsize=12)
        ax.legend(fontsize=7, ncol=3)
        plt.tight_layout()

        if save:
            path = os.path.join(self.results_dir, "distance_distribution.png")
            plt.savefig(path, dpi=150)
            print(f"[图表] 距离分布图已保存: {path}")
        plt.show()
        plt.close()


    def plot_tsne(self, test_results: dict, save: bool = True):
        """
        t-SNE 特征空间可视化（最多使用 2000 个样本避免太慢）。

        图例区分:
        ○ 已知目标类 (蓝色系，按类别区分颜色)
        □ 已知负类   (橙色系，按类别区分颜色)
        △ 真未知类   (绿色系，按类别区分颜色)
        ★ 球心 (与对应目标类同色)
        """
        feats = test_results["feats"]
        labels = test_results["labels"]
        cls_names_arr = np.array(test_results["cls_names"])

        # 降采样
        max_samples = 2000
        if len(feats) > max_samples:
            idx = np.random.choice(len(feats), max_samples, replace=False)
            feats = feats[idx]
            labels = labels[idx]
            cls_names_arr = cls_names_arr[idx]

        # 将球心拼接进去，保证同一坐标系
        centers = self.model.centers.detach().cpu().numpy()  # (K, D)
        combined = np.concatenate([feats, centers], axis=0)

        print(f"[t-SNE] 正在计算 {len(feats)} 个样本 + {len(centers)} 个球心的 t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                    random_state=42)
        emb_combined = tsne.fit_transform(combined)

        emb = emb_combined[:len(feats)]
        emb_centers = emb_combined[len(feats):]

        colors_known = plt.cm.Blues(np.linspace(0.4, 0.9, max(self.num_classes, 1)))
        colors_kn = plt.cm.Oranges(np.linspace(0.4, 0.9, max(len(self.known_negative_classes), 1)))
        colors_unk = plt.cm.Greens(np.linspace(0.4, 0.9, max(len(self.unknown_classes), 1)))

        fig, ax = plt.subplots(figsize=(10, 8))

        # ── 已知目标类 ──
        for c, name in enumerate(self.known_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                    c=[colors_known[c]], label=name,
                    alpha=0.6, s=20, marker="o")

        # ── 已知负类 ──
        for i, name in enumerate(self.known_negative_classes):
            mask = (cls_names_arr == name)
            if mask.sum() == 0:
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                    c=[colors_kn[i]], label=f"{name} (已知负类)",
                    alpha=0.6, s=20, marker="s")

        # ── 真未知类 ──
        for i, name in enumerate(self.unknown_classes):
            mask = (cls_names_arr == name)
            if mask.sum() == 0:
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1],
                    c=[colors_unk[i]], label=f"{name} (真未知类)",
                    alpha=0.6, s=20, marker="^")

        # ── 叠加球心 ──
        for c, name in enumerate(self.known_classes):
            ax.scatter(emb_centers[c, 0], emb_centers[c, 1],
                    c=[colors_known[c]], marker="*", s=300,
                    edgecolors="black", linewidths=1.2,
                    label=f"{name} 球心")

        ax.set_title("t-SNE 特征空间可视化\n"
                    "(○=已知目标类, □=已知负类, △=真未知类, ★=球心)")
        ax.legend(fontsize=7, markerscale=1.2, ncol=2,
                loc="upper left", bbox_to_anchor=(1.02, 1.0))
        plt.tight_layout()

        if save:
            path = os.path.join(self.results_dir, "tsne_features.png")
            plt.savefig(path, dpi=150)
            print(f"[图表] t-SNE 图已保存: {path}")
        plt.show()
        plt.close()