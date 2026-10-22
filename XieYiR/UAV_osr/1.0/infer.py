"""
infer.py
单信号/批量推理入口（多球体 Deep SVDD 版本）

用法:
  python infer.py --signal data/raw/sample.npy \
                  --model models/best_model.pth \
                  --config configs/config.yaml

输入格式:
  --signal 指向一个 .npy 文件，支持两种内容:
    1. 纯数组: shape (1024,) 单条信号 或 (N, 1024) 多条信号
    2. 字典: {"data": (N, 1024), ...}（与训练数据格式兼容，"labels"字段会被忽略）

输出示例:
  ════════════════════════════════════════
  推理结果
  ════════════════════════════════════════
  文件          : sample.npy
  片段数        : 20
  投票分布      : DJI_M600=15, DJI_PHANTOM=3, unknown=2
  最终判定      : DJI_M600 (已知目标类)
  最近球心距离  : 均值=0.4123  最小=0.2011
  判定阈值      : 0.8500 (类别: DJI_M600)
  置信度        : 高
  ════════════════════════════════════════
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, get_device
from src.model import MultiSphereDeepSVDDModel
from src.dataset import normalize_signal


def parse_args():
    parser = argparse.ArgumentParser(
        description="多目标开放集识别 - 单信号/批量推理"
    )
    parser.add_argument("--signal", type=str, required=True,
                        help="输入信号文件路径 (.npy)")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--threshold-file", type=str, default=None,
                        help="指定 metrics.json 路径以覆盖模型内置阈值 "
                             "(默认使用 checkpoint 中保存的 model.thresholds)")
    parser.add_argument("--threshold-scale", type=float, default=1.0,
                        help="对所有类别阈值统一做缩放，用于临时调整判定的宽严 "
                             "(例如设为1.2表示放宽5%%阈值)")
    return parser.parse_args()


def _load_signal_array(signal_path: str) -> np.ndarray:
    """
    加载输入信号，兼容纯数组和字典两种格式。

    Returns:
        (N, 1024) float32 数组
    """
    raw = np.load(signal_path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()

    if isinstance(raw, dict):
        for key in ["data", "X", "signals", "samples"]:
            if key in raw:
                arr = np.asarray(raw[key])
                break
        else:
            raise KeyError(f"字典中未找到数据字段，实际键: {list(raw.keys())}")
    else:
        arr = np.asarray(raw)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    return arr.astype(np.float32)


def _load_thresholds(model: MultiSphereDeepSVDDModel,
                     threshold_file: str = None,
                     device: torch.device = None) -> np.ndarray:
    """
    获取逐类阈值，优先级:
      1. 显式指定的 --threshold-file (metrics.json)
      2. checkpoint 中随模型一起保存的 model.thresholds buffer
    """
    if threshold_file is not None and os.path.exists(threshold_file):
        with open(threshold_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if "thresholds" not in metrics:
            raise KeyError(f"{threshold_file} 中未找到 'thresholds' 字段")
        thresholds = np.array(metrics["thresholds"], dtype=np.float32)
        print(f"[Infer] 阈值来源: {threshold_file}")
    else:
        thresholds = model.thresholds.detach().cpu().numpy()
        print(f"[Infer] 阈值来源: 模型 checkpoint 内置 thresholds")

    if np.isinf(thresholds).any():
        print(f"[警告] 部分类别阈值为 inf（可能从未标定过），"
              f"这些类别将永远不会被拒识: {thresholds}")

    return thresholds


def infer_signal(signal_path: str,
                 model: MultiSphereDeepSVDDModel,
                 cfg: dict,
                 device: torch.device,
                 thresholds: np.ndarray,
                 threshold_scale: float = 1.0) -> dict:
    """
    对单个 .npy 文件中的一条或多条信号进行推理。

    Returns:
        dict: 包含每类投票数、最终判定类别、距离统计、置信度等信息
    """
    feat_cfg = cfg.get("feature", {})
    normalize_method = feat_cfg.get("normalize_signal", "zscore")
    target_classes = cfg["data"]["target_classes"]

    raw_arr = _load_signal_array(signal_path)   # (N, 1024)
    n_segments = raw_arr.shape[0]

    norm_arr = np.stack([
        normalize_signal(row, method=normalize_method) for row in raw_arr
    ], axis=0)

    x = torch.from_numpy(norm_arr).unsqueeze(1).float().to(device)  # (N,1,1024)

    scaled_thresholds = torch.tensor(
        thresholds * threshold_scale, device=device, dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        feat = model(x)
        dist_sq_all = model.compute_distances(feat)   # (N, K)
        dist_all = dist_sq_all.sqrt()

        min_dist, nearest_class = dist_all.min(dim=1)  # (N,), (N,)
        class_threshold = scaled_thresholds[nearest_class]
        is_known = min_dist < class_threshold
        pred_class = torch.where(
            is_known, nearest_class,
            torch.full_like(nearest_class, -1)
        )

    min_dist_np = min_dist.cpu().numpy()
    nearest_class_np = nearest_class.cpu().numpy()
    pred_class_np = pred_class.cpu().numpy()

    # ── 逐段投票统计 ──
    vote_counts = {}
    for c in range(model.num_classes):
        vote_counts[target_classes[c]] = int((pred_class_np == c).sum())
    vote_counts["unknown"] = int((pred_class_np == -1).sum())

    # ── 多数投票决定最终判定 ──
    final_key = max(vote_counts, key=vote_counts.get)
    final_is_unknown = (final_key == "unknown")
    final_class_idx = -1 if final_is_unknown else target_classes.index(final_key)

    # ── 该最终判定类别对应的统计信息 ──
    if final_is_unknown:
        # 未知情况下，参考"投票给unknown"的那些段的最近球心距离
        mask = (pred_class_np == -1)
        relevant_threshold = None
    else:
        mask = (pred_class_np == final_class_idx)
        relevant_threshold = float(thresholds[final_class_idx] * threshold_scale)

    if mask.sum() > 0:
        dist_mean = float(min_dist_np[mask].mean())
        dist_min = float(min_dist_np[mask].min())
    else:
        dist_mean = float(min_dist_np.mean())
        dist_min = float(min_dist_np.min())

    # ── 置信度评估 ──
    if final_is_unknown:
        confidence = "高（多数片段被判定为未知信号）"
    else:
        ratio = dist_mean / (relevant_threshold + 1e-8)
        if ratio < 0.5:
            confidence = "极高"
        elif ratio < 0.8:
            confidence = "高"
        elif ratio < 1.0:
            confidence = "中等"
        else:
            confidence = "低（存在争议，建议复核）"

    return {
        "n_segments": n_segments,
        "vote_counts": vote_counts,
        "final_key": final_key,
        "final_is_unknown": final_is_unknown,
        "dist_mean": dist_mean,
        "dist_min": dist_min,
        "relevant_threshold": relevant_threshold,
        "confidence": confidence,
        "per_segment_pred": pred_class_np.tolist(),
        "per_segment_dist": min_dist_np.tolist(),
        "per_segment_nearest_class": nearest_class_np.tolist()
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    # ── 加载模型 ──
    model = MultiSphereDeepSVDDModel(cfg).to(device)
    if os.path.exists(args.model):
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Infer] 模型已加载: {args.model}")
    else:
        print(f"[警告] 模型文件不存在: {args.model}")
        return

    thresholds = _load_thresholds(model, args.threshold_file, device)

    # ── 推理 ──
    result = infer_signal(
        signal_path=args.signal,
        model=model,
        cfg=cfg,
        device=device,
        thresholds=thresholds,
        threshold_scale=args.threshold_scale
    )

    # ── 输出报告 ──
    fname = os.path.basename(args.signal)
    vote_str = ", ".join(f"{k}={v}" for k, v in result["vote_counts"].items())

    if result["final_is_unknown"]:
        decision = "❌ 判定为未知信号（拒识）"
        threshold_str = "N/A（无匹配已知类）"
    else:
        decision = f"✅ 判定为已知目标类: {result['final_key']}"
        threshold_str = f"{result['relevant_threshold']:.4f} (类别: {result['final_key']})"

    print("\n" + "═"*44)
    print("  推理结果")
    print("═"*44)
    print(f"  文件          : {fname}")
    print(f"  片段数        : {result['n_segments']}")
    print(f"  投票分布      : {vote_str}")
    print(f"  最终判定      : {decision}")
    print(f"  最近球心距离  : 均值={result['dist_mean']:.4f}  "
          f"最小={result['dist_min']:.4f}")
    print(f"  判定阈值      : {threshold_str}")
    print(f"  置信度        : {result['confidence']}")
    print("═"*44 + "\n")


if __name__ == "__main__":
    main()