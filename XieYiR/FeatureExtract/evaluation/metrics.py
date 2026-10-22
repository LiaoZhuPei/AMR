# evaluation/metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from typing import List, Dict

class OpenSetEvaluator:
    """
    开放集检测评估
    关键指标:
    - AUROC: 整体区分能力
    - AUPR:  在类别不平衡下更有意义
    - FPR@95TPR: 95% TPR时的误报率（实际应用关注）
    - 各SNR下的性能
    """
    
    def evaluate(self, y_true: np.ndarray, 
                  scores: np.ndarray,
                  meta_modes: List[str] = None,
                  meta_snr: List[int] = None,
                  plot: bool = True) -> Dict:
        """
        全面评估开放集检测性能
        y_true: +1 (Link11), -1 (Unknown)
        scores: 融合得分（越高越像Link11）
        """
        results = {}
        
        # 二值标签（1=正类，0=负类）用于sklearn
        y_bin = (y_true == 1).astype(int)
        
        # ---- 1. AUROC & AUPR ----
        auroc = roc_auc_score(y_bin, scores)
        aupr  = average_precision_score(y_bin, scores)
        results['AUROC'] = auroc
        results['AUPR']  = aupr
        
        # ---- 2. FPR@95TPR ----
        fpr, tpr, thresholds = roc_curve(y_bin, scores)
        idx_95 = np.searchsorted(tpr, 0.95)
        fpr_at_95tpr = fpr[min(idx_95, len(fpr)-1)]
        results['FPR@95TPR'] = fpr_at_95tpr
        
        # ---- 3. 最优阈值（F1最大化） ----
        prec, rec, pr_thresholds = precision_recall_curve(y_bin, scores)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx  = np.argmax(f1_scores)
        best_thresh = pr_thresholds[min(best_idx, len(pr_thresholds)-1)]
        best_f1     = f1_scores[best_idx]
        results['Best_Threshold'] = best_thresh
        results['Best_F1']        = best_f1
        
        # ---- 4. 混淆矩阵（使用最优阈值） ----
        y_pred = (scores >= best_thresh).astype(int) * 2 - 1
        cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
        results['confusion_matrix'] = cm
        
        print("\n" + "="*60)
        print("         开放集检测评估结果")
        print("="*60)
        print(f"  AUROC:        {auroc:.4f}")
        print(f"  AUPR:         {aupr:.4f}")
        print(f"  FPR@95TPR:    {fpr_at_95tpr:.4f}")
        print(f"  最优F1:       {best_f1:.4f}  (阈值={best_thresh:.4f})")
        print("\n  混淆矩阵 (行=真实, 列=预测):")
        print(f"  [TP={cm[0,0]}, FN={cm[0,1]}]")
        print(f"  [FP={cm[1,0]}, TN={cm[1,1]}]")
        
        # ---- 5. 按SNR分析 ----
        if meta_snr is not None:
            snr_results = {}
            unique_snrs = sorted(set(meta_snr), reverse=True)
            print("\n  各SNR下的AUROC:")
            for snr in unique_snrs:
                mask = np.array(meta_snr) == snr
                if y_bin[mask].sum() == 0 or (1 - y_bin[mask]).sum() == 0:
                    continue
                snr_auroc = roc_auc_score(y_bin[mask], scores[mask])
                snr_results[snr] = snr_auroc
                print(f"    SNR={snr:4d}dB: AUROC={snr_auroc:.4f}")
            results['snr_auroc'] = snr_results
        
        # ---- 6. 按信号类型分析 ----
        if meta_modes is not None:
            mode_results = {}
            unique_modes = set(meta_modes)
            print("\n  各信号类型检测率:")
            for mode in sorted(unique_modes):
                mask = np.array(meta_modes) == mode
                mode_scores = scores[mask]
                if y_true[mask][0] == 1:  # 正类
                    rate = np.mean(mode_scores >= best_thresh)
                    print(f"    {mode:20s}: TPR={rate:.4f}")
                else:  # 负类（应该被拒绝）
                    rate = np.mean(mode_scores < best_thresh)
                    print(f"    {mode:20s}: TNR={rate:.4f}")
                mode_results[mode] = rate
            results['mode_results'] = mode_results
        
        if plot:
            self._plot_results(y_bin, scores, fpr, tpr, 
                               prec, rec, results)
        
        return results
    
    def _plot_results(self, y_bin, scores, fpr, tpr, 
                       prec, rec, results):
        """绘制评估图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # ROC曲线
        axes[0].plot(fpr, tpr, 'b-', linewidth=2,
                     label=f'AUROC={results["AUROC"]:.4f}')
        axes[0].plot([0, 1], [0, 1], 'k--')
        axes[0].axvline(x=results['FPR@95TPR'], color='r', 
                        linestyle='--', label=f'FPR@95TPR={results["FPR@95TPR"]:.4f}')
        axes[0].set_xlabel('FPR')
        axes[0].set_ylabel('TPR')
        axes[0].set_title('ROC曲线')
        axes[0].legend()
        axes[0].grid(True)
        
        # PR曲线
        axes[1].plot(rec, prec, 'g-', linewidth=2,
                     label=f'AUPR={results["AUPR"]:.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('PR曲线')
        axes[1].legend()
        axes[1].grid(True)
        
        # 得分分布
        pos_scores = scores[y_bin == 1]
        neg_scores = scores[y_bin == 0]
        axes[2].hist(pos_scores, bins=50, alpha=0.6, 
                     color='blue', label='Link11', density=True)
        axes[2].hist(neg_scores, bins=50, alpha=0.6, 
                     color='red', label='Unknown', density=True)
        axes[2].axvline(x=results['Best_Threshold'], color='k',
                        linestyle='--', label=f'阈值={results["Best_Threshold"]:.3f}')
        axes[2].set_xlabel('融合得分')
        axes[2].set_ylabel('密度')
        axes[2].set_title('得分分布')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("[INFO] 图表已保存至 results/evaluation.png")