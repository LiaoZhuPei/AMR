"""
infer.py
单信号推理入口

用法:
  python infer.py --signal data/raw/MT63/test_signal.wav \
                  --model models/best_model.pth \
                  --config configs/config.yaml

输出示例:
  ══════════════════════════════
  推理结果
  ══════════════════════════════
  文件: test_signal.wav
  距离: 0.3821
  阈值: 0.8500
  决策: ✅ 判定为 Link11（合法信号）
  置信度: 高（距离远低于阈值）
  ══════════════════════════════
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, get_device
from src.model import DeepSVDDModel
from src.dataset import iq_to_spectrogram


def parse_args():
    parser = argparse.ArgumentParser(
        description="Link11 开放集识别 - 单信号推理"
    )
    parser.add_argument("--signal", type=str, required=True,
                        help="输入信号文件路径 (.wav)")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--threshold", type=float, default=None,
                        help="手动指定阈值（覆盖模型内置阈值）")
    return parser.parse_args()


def infer_signal(signal_path: str,
                 model: DeepSVDDModel,
                 cfg: dict,
                 device: torch.device,
                 threshold_override: float = None) -> dict:
    """
    对单个信号文件进行推理。

    Returns:
        dict: 包含 dist_mean, is_target, confidence, segment_results
    """
    data_cfg = cfg["data"]
    feat_cfg = cfg["feature"]

    sample_rate = data_cfg["sample_rate"]
    segment_len = int(data_cfg["segment_duration"] * sample_rate)
    hop_len = int(segment_len * (1 - data_cfg["overlap"]))

    # 加载信号
    raw = np.load(signal_path)
    specs = []
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    for row in raw:
        spec = iq_to_spectrogram(row, n_fft=feat_cfg["n_fft"], hop_length=feat_cfg["hop_length"], normalize=feat_cfg["normalize"])
        specs.append(spec)

    x = torch.from_numpy(np.stack(specs)).unsqueeze(1).float().to(device)

    # 推理
    model.eval()
    with torch.no_grad():
        feat = model(x)
        dist_sq = model.compute_distances(feat)
        dists = dist_sq.sqrt().cpu().numpy()

    with open("results/metrics.json", "r") as f:
        metrics = json.load(f)

    threshold_override = metrics["threshold"]

    # 阈值
    threshold = (threshold_override
                 if threshold_override is not None
                 else model.threshold.item())

    dist_mean = float(dists.mean())
    dist_min = float(dists.min())

    # 多数投票决策
    votes_link11 = int((dists < threshold).sum())
    total_votes = len(dists)
    is_target = votes_link11 > total_votes // 2

    # 置信度（基于距离与阈值的比值）
    ratio = dist_mean / (threshold + 1e-8)
    if is_target:
        if ratio < 0.5:
            confidence = "极高"
        elif ratio < 0.8:
            confidence = "高"
        else:
            confidence = "中等"
    else:
        if ratio > 2.0:
            confidence = "极高（确定为未知信号）"
        elif ratio > 1.3:
            confidence = "高"
        else:
            confidence = "中等（距阈值较近，建议复核）"

    return {
        "dist_mean": dist_mean,
        "dist_min": dist_min,
        "threshold": threshold,
        "is_target": is_target,
        "confidence": confidence,
        "votes_link11": votes_link11,
        "total_segments": total_votes,
        "segment_dists": dists.tolist()
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    # 加载模型
    model = DeepSVDDModel(cfg).to(device)
    if os.path.exists(args.model):
        state = torch.load(args.model, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Infer] 模型已加载: {args.model}")
    else:
        print(f"[警告] 模型文件不存在: {args.model}")
        return

    # 推理
    result = infer_signal(
        signal_path=args.signal,
        model=model,
        cfg=cfg,
        device=device,
        threshold_override=args.threshold
    )

    # 输出报告
    fname = os.path.basename(args.signal)
    decision = "✅ 判定为 Link11（合法信号）" if result["is_target"] \
               else "❌ 判定为未知信号（拒识）"

    print("\n" + "═"*40)
    print("  推理结果")
    print("═"*40)
    print(f"  文件        : {fname}")
    print(f"  片段数      : {result['total_segments']}")
    print(f"  Link11票数  : {result['votes_link11']}/{result['total_segments']}")
    print(f"  平均距离    : {result['dist_mean']:.4f}")
    print(f"  决策阈值    : {result['threshold']:.4f}")
    print(f"  决策        : {decision}")
    print(f"  置信度      : {result['confidence']}")
    print("═"*40 + "\n")


if __name__ == "__main__":
    main()
