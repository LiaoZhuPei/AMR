"""
evaluator.py
评估模块

功能:
  1. 提取全量测试集特征 + 距离分数
  2. 阈值校准 (基于验证集 FPR 约束)
  3. 计算开放集评估指标:
     - AUROC
     - FPR@95TPR (在 95% 真阳率时的假阳率)
     - 开放集准确率 (已知类 / 未知类分开报告)
  4. 绘图:
     - 距离分布直方图 (正类 vs 负类)
     - ROC 曲线
     - t-SNE 特征可视化
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
    开放集识别评估器。

    Args:
        model:    DeepSVDDModel (已加载权重)
        device:   torch.device
        cfg:      完整配置字典
    """

    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.results_dir = cfg["output"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    #  特征提取
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, data_loader) -> dict:
        """
        从 DataLoader 中提取所有样本的特征和距离。

        Returns:
            dict with keys:
              "feats"     : np.ndarray (N, D)
              "dists"     : np.ndarray (N,)
              "labels"    : np.ndarray (N,) - 0:负类 / 1:正类
              "cls_names" : list[str] - 每个样本所属的原始类名
        """
        self.model.eval()
        all_feats, all_dists, all_labels, all_cls = [], [], [], []

        for x, labels, cls_names in data_loader:
            x = x.to(self.device)
            feat = self.model(x)
            dist_sq = self.model.compute_distances(feat)
            dist = dist_sq.sqrt()

            all_feats.append(feat.cpu().numpy())
            all_dists.append(dist.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cls.extend(cls_names)

        return {
            "feats": np.concatenate(all_feats, axis=0),
            "dists": np.concatenate(all_dists, axis=0),
            "labels": np.array(all_labels),
            "cls_names": all_cls
        }

    # ─────────────────────────────────────────────
    #  阈值校准
    # ─────────────────────────────────────────────

    def calibrate_threshold(self, val_results: dict) -> float:
        """
        在验证集上按目标 FPR 校准决策阈值。

        策略: 找到使正类 TPR ≥ 95% 时对应的最小距离阈值。
        即: 允许 5% 的正类被误判为未知，换取更低的假阳率。

        Args:
            val_results: extract_features() 的输出

        Returns:
            threshold: float
        """
        eval_cfg = self.cfg["evaluation"]
        strategy = eval_cfg["threshold_strategy"]
        dists = val_results["dists"]
        labels = val_results["labels"]

        if strategy == "val_fpr":
            # 正类距离的分布
            pos_dists = dists[labels == 1]
            if len(pos_dists) == 0:
                raise ValueError("No positive samples in validation set!")
            # 95th percentile of positive distances = 阈值
            # 即 95% 的正类会被判为 Link11
            threshold = np.percentile(pos_dists, 95)

        elif strategy == "percentile":
            pos_dists = dists[labels == 1]
            pct = eval_cfg.get("percentile", 95)
            threshold = np.percentile(pos_dists, pct)

        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

        print(f"[Evaluator] 阈值校准完成: {threshold:.4f} "
              f"(策略: {strategy})")

        # 写入模型
        self.model.threshold = torch.tensor(
            threshold, device=self.device
        )
        return threshold

    # ─────────────────────────────────────────────
    #  评估指标
    # ─────────────────────────────────────────────

    def compute_metrics(self, test_results: dict,
                        threshold: float) -> dict:
        """
        计算全套开放集评估指标。

        Args:
            test_results: extract_features() 输出
            threshold:    决策阈值

        Returns:
            metrics dict
        """
        dists = test_results["dists"]
        labels = test_results["labels"]
        cls_names = test_results["cls_names"]

        # 异常分数 = 距离（高 → 未知）
        scores = dists  # 高分 → 负类 / 未知

        # ── AUROC ──
        # 正类 label=1, 用 -scores 使正类得高分
        auroc = roc_auc_score(labels, -scores)

        # ── FPR@95TPR ──
        fpr_arr, tpr_arr, thresholds = roc_curve(labels, -scores)
        idx = np.argmin(np.abs(tpr_arr - 0.95))
        fpr_at_95tpr = fpr_arr[idx]

        # ── 决策结果 ──
        preds = (dists < threshold).astype(int)  # 1 → Link11

        # ── 分类型报告 ──
        target_cls = self.cfg["data"]["target_class"]
        known_neg = self.cfg["data"]["known_negatives"]
        unknown_neg = self.cfg["data"]["unknown_negatives"]

        def group_accuracy(cls_list, pred_label):
            """计算某组信号的判断准确率"""
            mask = np.array([c in cls_list for c in cls_names])
            if mask.sum() == 0:
                return None
            group_preds = preds[mask]
            correct = (group_preds == pred_label).sum()
            return correct / len(group_preds)

        metrics = {
            "auroc": auroc,
            "fpr_at_95tpr": fpr_at_95tpr,
            "threshold": threshold,
            "target_recall": group_accuracy([target_cls], 1),
            "known_neg_rejection_rate": group_accuracy(known_neg, 0),
            "unknown_neg_rejection_rate": group_accuracy(unknown_neg, 0),
            "roc_curve": (fpr_arr, tpr_arr)
        }

        return metrics

    # ─────────────────────────────────────────────
    #  打印报告
    # ─────────────────────────────────────────────

    def print_report(self, metrics: dict):
        print("\n" + "="*60)
        print("  开放集识别评估报告")
        print("="*60)
        print(f"  AUROC               : {metrics['auroc']:.4f}")
        print(f"  FPR@95TPR           : {metrics['fpr_at_95tpr']:.4f}")
        print(f"  决策阈值             : {metrics['threshold']:.4f}")
        print("-"*60)

        target_cls = self.cfg["data"]["target_class"]
        if metrics["target_recall"] is not None:
            print(f"  [{target_cls}] 正确识别率 (TPR)"
                  f"  : {metrics['target_recall']:.4f}  "
                  f"← 应接近 1.0")

        if metrics["known_neg_rejection_rate"] is not None:
            print(f"  [已知负类] 正确拒识率"
                  f"      : {metrics['known_neg_rejection_rate']:.4f}  "
                  f"← 应接近 1.0")

        if metrics["unknown_neg_rejection_rate"] is not None:
            print(f"  [未知负类] 开放集拒识率"
                  f"    : {metrics['unknown_neg_rejection_rate']:.4f}  "
                  f"← 关键指标，应接近 1.0")

        print("="*60 + "\n")

    # ─────────────────────────────────────────────
    #  可视化
    # ─────────────────────────────────────────────

    def plot_distance_distribution(self, test_results: dict,
                                   threshold: float,
                                   save: bool = True):
        """绘制各类信号的距离分布直方图"""
        dists = test_results["dists"]
        cls_names_arr = np.array(test_results["cls_names"])

        target_cls = self.cfg["data"]["target_class"]
        known_neg = self.cfg["data"]["known_negatives"]
        unknown_neg = self.cfg["data"]["unknown_negatives"]

        fig, ax = plt.subplots(figsize=(10, 5))

        # 正类
        mask_pos = (cls_names_arr == target_cls)
        if mask_pos.sum() > 0:
            ax.hist(dists[mask_pos], bins=50, alpha=0.7,
                    label=f"{target_cls} (Link11)", color="steelblue",
                    density=True)

        # 已知负类
        colors_known = ["tomato", "orange", "gold", "darkviolet"]
        for cls, c in zip(known_neg, colors_known):
            mask = (cls_names_arr == cls)
            if mask.sum() > 0:
                ax.hist(dists[mask], bins=50, alpha=0.6,
                        label=f"{cls} (已知负类)", color=c, density=True)

        # 未知负类
        colors_unknown = ["mediumorchid", "darkgreen", "sienna", "cadetblue", "darkcyan"]
        for cls, c in zip(unknown_neg, colors_unknown):
            mask = (cls_names_arr == cls)
            if mask.sum() > 0:
                ax.hist(dists[mask], bins=50, alpha=0.5,
                        label=f"{cls} (未知)", color=c,
                        linestyle="--", density=True)

        ax.axvline(threshold, color="black", linestyle="--",
                   linewidth=2, label=f"阈值={threshold:.3f}")

        ax.set_xlabel("到超球心的距离", fontsize=12)
        ax.set_ylabel("密度", fontsize=12)
        ax.set_title("各类信号距离分布（阈值左侧→判定为Link11）",
                     fontsize=13)
        ax.legend(fontsize=9)
        plt.tight_layout()

        if save:
            path = os.path.join(self.results_dir, "distance_distribution.png")
            plt.savefig(path, dpi=150)
            print(f"[图表] 距离分布图已保存: {path}")
        plt.show()
        plt.close()

    def plot_roc_curve(self, metrics: dict, save: bool = True):
        """绘制 ROC 曲线"""
        fpr, tpr = metrics["roc_curve"]
        auroc = metrics["auroc"]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="steelblue", lw=2,
                label=f"ROC Curve (AUROC = {auroc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.axhline(0.95, color="gray", linestyle=":", lw=1,
                   label="TPR = 0.95")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Link11 vs. Non-Link11")
        ax.legend()
        plt.tight_layout()

        if save:
            path = os.path.join(self.results_dir, "roc_curve.png")
            plt.savefig(path, dpi=150)
            print(f"[图表] ROC 曲线已保存: {path}")
        plt.show()
        plt.close()

    def plot_tsne(self, test_results: dict, save: bool = True):
        """
        t-SNE 特征空间可视化（最多使用 2000 个样本避免太慢）
        """
        feats = test_results["feats"]
        cls_names_arr = np.array(test_results["cls_names"])

        # 降采样
        max_samples = 2000
        if len(feats) > max_samples:
            idx = np.random.choice(len(feats), max_samples, replace=False)
            feats = feats[idx]
            cls_names_arr = cls_names_arr[idx]

        print(f"[t-SNE] 正在计算 {len(feats)} 个样本的 t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                    random_state=42)
        emb = tsne.fit_transform(feats)

        # 类别颜色映射
        target_cls = self.cfg["data"]["target_class"]
        known_neg = self.cfg["data"]["known_negatives"]
        unknown_neg = self.cfg["data"]["unknown_negatives"]

        color_map = {target_cls: "steelblue"}
        for cls, c in zip(known_neg, ["tomato", "orange", "gold","darkviolet"]):
            color_map[cls] = c
        for cls, c in zip(unknown_neg,
                          ["mediumorchid", "darkgreen", "sienna", "cadetblue", "darkcyan"]):
            color_map[cls] = c

        fig, ax = plt.subplots(figsize=(9, 7))
        for cls_name, color in color_map.items():
            mask = (cls_names_arr == cls_name)
            if mask.sum() == 0:
                continue
            marker = "o" if cls_name == target_cls else (
                "s" if cls_name in known_neg else "^"
            )
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, label=cls_name,
                       alpha=0.6, s=20, marker=marker)

        ax.set_title("t-SNE 特征空间可视化\n"
                     "(○=Link11, □=已知负类, △=未知负类)")
        ax.legend(fontsize=9, markerscale=1.5)
        plt.tight_layout()

        if save:
            path = os.path.join(self.results_dir, "tsne_features.png")
            plt.savefig(path, dpi=150)
            print(f"[图表] t-SNE 图已保存: {path}")
        plt.show()
        plt.close()
