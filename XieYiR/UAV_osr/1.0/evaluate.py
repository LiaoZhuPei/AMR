"""
evaluate.py
评估入口（多球体 Deep SVDD 版本）

用法:
  python evaluate.py --config configs/config.yaml
  python evaluate.py --config configs/config.yaml --synthetic
  python evaluate.py --config configs/config.yaml --model models/best_model.pth
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# 解决保存图像是负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_seed, get_device
from src.utils import generate_synthetic_data
from src.model import MultiSphereDeepSVDDModel
from src.evaluator import Evaluator
from src.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="多目标开放集识别 - 评估脚本 (Multi-Sphere Deep SVDD)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成数据")
    parser.add_argument("--no-plot", action="store_true",
                        help="跳过绘图（无 GUI 环境）")
    return parser.parse_args()


def _validate_cfg(cfg: dict):
    """多球体模型关键配置一致性检查"""
    num_classes = cfg["model"]["num_classes"]
    target_classes = cfg["data"]["target_classes"]
    if num_classes != len(target_classes):
        raise ValueError(
            f"[配置错误] model.num_classes({num_classes}) 与 "
            f"data.target_classes 长度({len(target_classes)}) 不一致！"
        )


def _make_json_serializable(obj):
    """
    递归将 metrics 字典中的各种类型（ndarray、dict、numpy标量、None）
    转换为可以被 json.dump 序列化的形式，避免新增的复杂指标被静默丢弃。
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    # 其他不可序列化类型（如 roc_curve 的元组数组已在上面处理），兜底转字符串
    return str(obj)


def main():
    args = parse_args()
    setup_logging()

    cfg = load_config(args.config)
    _validate_cfg(cfg)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()

    # ── 数据 ──
    if args.synthetic:
        train_loader, val_loader, test_loader = generate_synthetic_data(cfg)
    else:
        train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── 加载模型 ──
    model = MultiSphereDeepSVDDModel(cfg).to(device)
    if os.path.exists(args.model):
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Evaluate] 模型已加载: {args.model}")
    else:
        print(f"[警告] 模型文件不存在: {args.model}，使用随机初始化权重（仅调试）")

    print(f"[Evaluate] 已知类别数(K): {model.num_classes} | "
          f"目标类: {cfg['data']['target_classes']}")

    evaluator = Evaluator(model, device, cfg)

    # ── 阈值校准（用验证集，验证集中应只含已知类样本，label>=0）──
    print("\n[Step 1] 在验证集上提取特征，并按类别校准阈值...")
    val_results = evaluator.extract_features(val_loader)
    thresholds = evaluator.calibrate_threshold(val_results)

    # ── 测试集评估（含已知负类 + 未知类，label=-1）──
    print("\n[Step 2] 在测试集上评估（含未知类）...")
    test_results = evaluator.extract_features(test_loader)
    metrics = evaluator.compute_metrics(test_results, thresholds)

    # ── 报告 ──
    evaluator.print_report(metrics)

    # ── 可视化 ──
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")  # 无头模式，保存为文件
        evaluator.plot_distance_distribution(test_results, thresholds)
        evaluator.plot_roc_curve(metrics)
        evaluator.plot_tsne(test_results)
        print(f"\n[输出] 所有图表已保存到 {cfg['output']['results_dir']}/")

    # ── 保存指标 ──
    # 排除掉不适合/不需要落盘的大体积或重复字段（preds 数组、roc_curve 原始点）
    metrics_to_save = {
        k: v for k, v in metrics.items()
        if k not in ("roc_curve", "preds")
    }
    metrics_to_save = _make_json_serializable(metrics_to_save)

    out_dir = cfg["output"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    print(f"[输出] 指标已保存: {out_path}")


if __name__ == "__main__":
    main()