"""
smoke_test.py
端到端冒烟测试（无需真实数据）

功能:
  1. 用合成数据跑完整训练 + 评估 + 推理流程
  2. 验证所有模块能协同工作
  3. 输出简洁的通过/失败报告

运行:
  python smoke_test.py
"""

import os
import sys
import json
import shutil
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # 无头模式
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 解决保存图像是负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, set_seed, get_device
from src.utils import generate_synthetic_data, count_parameters
from src.model import DeepSVDDModel, SVDDLoss, ContrastiveLoss
from src.trainer import Trainer
from src.evaluator import Evaluator

logging.basicConfig(level=logging.WARNING)  # 测试时静默大部分日志


def run_smoke_test():
    print("\n" + "="*60)
    print("  Link11 OSR - 端到端冒烟测试")
    print("="*60)

    # ─── 配置 ───
    cfg = load_config("configs/config.yaml")
    # 快速测试用的小参数
    cfg["training"]["epochs"] = 8
    cfg["training"]["batch_size"] = 32
    cfg["training"]["svdd"]["warm_up_epochs"] = 2
    cfg["training"]["patience"] = 999
    cfg["model"]["feature_dim"] = 64
    cfg["output"]["model_dir"] = "models/_smoke"
    cfg["output"]["results_dir"] = "results/_smoke"
    os.makedirs(cfg["output"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)

    set_seed(42)
    device = get_device()

    results = {}

    # ─── Test 1: 合成数据生成 ───
    print("\n[Test 1] 合成数据生成...", end=" ")
    try:
        train_loader, val_loader, test_loader = generate_synthetic_data(cfg)
        x, y, cls = next(iter(train_loader))
        assert x.shape[1] == 1, "频谱图应有1个通道"
        assert x.dtype == torch.float32
        print(f"✓  (批次形状: {tuple(x.shape)}, 类型: {set(cls)})")
        results["数据生成"] = "通过"
    except Exception as e:
        print(f"✗  ({e})")
        results["数据生成"] = f"失败: {e}"
        return results

    # ─── Test 2: 模型构建 ───
    print("[Test 2] 模型构建...", end=" ")
    try:
        model = DeepSVDDModel(cfg).to(device)
        n_params = count_parameters(model)
        feat = model(x.to(device))
        assert feat.shape == (x.shape[0], cfg["model"]["feature_dim"])
        dist_sq = model.compute_distances(feat)
        assert dist_sq.shape == (x.shape[0],)
        print(f"✓  (参数: {n_params:,}, 特征: {tuple(feat.shape)})")
        results["模型构建"] = "通过"
    except Exception as e:
        print(f"✗  ({e})")
        results["模型构建"] = f"失败: {e}"
        return results

    # ─── Test 3: 损失函数 ───
    print("[Test 3] 损失函数...", end=" ")
    try:
        svdd_loss = SVDDLoss(nu=0.1)
        contrast_loss = ContrastiveLoss(margin=2.0)
        loss1 = svdd_loss(dist_sq)
        loss2 = contrast_loss(feat, y.to(device))
        assert loss1.item() >= 0
        assert loss2.item() >= 0
        print(f"✓  (SVDD={loss1.item():.4f}, Contrast={loss2.item():.4f})")
        results["损失函数"] = "通过"
    except Exception as e:
        print(f"✗  ({e})")
        results["损失函数"] = f"失败: {e}"

    # ─── Test 4: 训练流程 ───
    print("[Test 4] 训练流程...", end=" ")
    try:
        trainer = Trainer(model, train_loader, val_loader, cfg, device)
        trainer.train()
        assert os.path.exists(
            os.path.join(cfg["output"]["model_dir"], "best_model.pth")
        )
        print(f"✓  (最佳 AUROC: {trainer.best_auroc:.4f})")
        results["训练流程"] = f"通过 (AUROC={trainer.best_auroc:.4f})"
    except Exception as e:
        print(f"✗  ({e})")
        results["训练流程"] = f"失败: {e}"

    # ─── Test 5: 评估流程 ───
    print("[Test 5] 评估流程...", end=" ")
    try:
        evaluator = Evaluator(model, device, cfg)
        val_results = evaluator.extract_features(val_loader)
        threshold = evaluator.calibrate_threshold(val_results)
        test_results = evaluator.extract_features(test_loader)
        metrics = evaluator.compute_metrics(test_results, threshold)

        assert "auroc" in metrics
        assert 0.0 <= metrics["auroc"] <= 1.0
        print(f"✓  (AUROC={metrics['auroc']:.4f}, "
              f"FPR@95TPR={metrics['fpr_at_95tpr']:.4f})")
        results["评估流程"] = f"通过 (AUROC={metrics['auroc']:.4f})"
    except Exception as e:
        print(f"✗  ({e})")
        results["评估流程"] = f"失败: {e}"

    # ─── Test 6: 可视化 ───
    print("[Test 6] 可视化输出...", end=" ")
    try:
        evaluator.plot_distance_distribution(test_results, threshold)
        evaluator.plot_roc_curve(metrics)
        evaluator.plot_tsne(test_results)
        plots = os.listdir(cfg["output"]["results_dir"])
        print(f"✓  (已生成: {plots})")
        results["可视化"] = "通过"
    except Exception as e:
        print(f"✗  ({e})")
        results["可视化"] = f"失败: {e}"

    # ─── Test 7: 单信号推理 ───
    print("[Test 7] 单信号推理...", end=" ")
    try:
        from infer import infer_signal
        # 用合成数据的第一个批次第一个样本模拟推理
        # 这里直接测试推理函数的核心逻辑
        test_x, test_y, test_cls = next(iter(test_loader))
        model.eval()
        with torch.no_grad():
            feat_t = model(test_x.to(device))
            dists = model.compute_distances(feat_t).sqrt()
            preds = (dists < threshold).cpu().numpy()
            true_labels = test_y.numpy()

        pos_acc = preds[true_labels == 1].mean()
        neg_acc = (1 - preds[true_labels == 0]).mean()
        print(f"✓  (正类召回: {pos_acc:.2f}, 负类拒识: {neg_acc:.2f})")
        results["推理流程"] = "通过"
    except Exception as e:
        print(f"✗  ({e})")
        results["推理流程"] = f"失败: {e}"

    # ─── 汇总 ───
    print("\n" + "─"*60)
    print("  测试汇总")
    print("─"*60)
    all_pass = True
    for name, result in results.items():
        icon = "✓" if result.startswith("通过") else "✗"
        print(f"  {icon}  {name}: {result}")
        if not result.startswith("通过"):
            all_pass = False

    print("─"*60)
    if all_pass:
        print("  🎉 所有测试通过！项目框架可正常运行。")
        print("  下一步: 准备真实数据放入 data/raw/ 后运行 train.py")
    else:
        print("  ⚠️  部分测试失败，请检查上方错误信息。")
    print("="*60 + "\n")

    # 清理临时文件
    shutil.rmtree("models/_smoke", ignore_errors=True)
    shutil.rmtree("results/_smoke", ignore_errors=True)

    return results


if __name__ == "__main__":
    run_smoke_test()
