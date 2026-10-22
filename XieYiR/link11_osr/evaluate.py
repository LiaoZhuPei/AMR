"""
evaluate.py
评估入口

用法:
  python evaluate.py --config configs/config.yaml
  python evaluate.py --config configs/config.yaml --synthetic
  python evaluate.py --config configs/config.yaml --model models/best_model.pth
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 解决保存图像是负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_seed, get_device
from src.utils import generate_synthetic_data
from src.model import DeepSVDDModel
from src.evaluator import Evaluator
from src.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Link11 开放集识别 - 评估脚本"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成数据")
    parser.add_argument("--no-plot", action="store_true",
                        help="跳过绘图（无 GUI 环境）")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()

    # ── 数据 ──
    if args.synthetic:
        train_loader, val_loader, test_loader = generate_synthetic_data(cfg)
    else:
        train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── 加载模型 ──
    model = DeepSVDDModel(cfg).to(device)
    if os.path.exists(args.model):
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Evaluate] 模型已加载: {args.model}")
    else:
        print(f"[警告] 模型文件不存在: {args.model}，使用随机初始化权重（仅调试）")

    evaluator = Evaluator(model, device, cfg)

    # ── 阈值校准（用验证集）──
    print("\n[Step 1] 在验证集上提取特征...")
    val_results = evaluator.extract_features(val_loader)
    threshold = evaluator.calibrate_threshold(val_results)

    # ── 测试集评估 ──
    print("\n[Step 2] 在测试集上评估（含未知类）...")
    test_results = evaluator.extract_features(test_loader)
    # metrics = evaluator.compute_metrics(val_results, threshold + 0.1)
    metrics = evaluator.compute_metrics(test_results, threshold + 0.1)

    # ── 报告 ──
    evaluator.print_report(metrics)

    # ── 可视化 ──
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")  # 无头模式，保存为文件
        evaluator.plot_distance_distribution(test_results, threshold)
        evaluator.plot_roc_curve(metrics)
        evaluator.plot_tsne(test_results)
        print(f"\n[输出] 所有图表已保存到 {cfg['output']['results_dir']}/")

    # ── 保存指标 ──
    import json
    metrics_to_save = {
        k: (float(v) if isinstance(v, (float, int)) else None)
        for k, v in metrics.items()
        if k != "roc_curve"
    }
    out_path = os.path.join(cfg["output"]["results_dir"], "metrics.json")
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"[输出] 指标已保存: {out_path}")


if __name__ == "__main__":
    main()
