"""
train.py
训练入口

用法:
  # 使用真实数据（需先准备好 data/raw/）
  python train.py --config configs/config.yaml

  # 使用合成数据（无需真实数据，快速验证流程）
  python train.py --config configs/config.yaml --synthetic
"""

import argparse
import logging
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_seed, get_device
from src.utils import generate_synthetic_data, count_parameters
from src.model import DeepSVDDModel
from src.trainer import Trainer
from src.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="Link11 开放集识别 - 训练脚本"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="使用合成数据（用于调试，无需真实数据集）"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="强制指定设备 (cpu/cuda/mps)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 初始化 ──
    setup_logging(level=logging.INFO,
                  log_file="results/train.log")
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])

    if args.device:
        import torch
        device = torch.device(args.device)
        print(f"[Device] 手动指定: {device}")
    else:
        device = get_device()

    # ── 数据加载 ──
    if args.synthetic:
        print("\n[模式] 使用合成数据（验证流程）")
        train_loader, val_loader, test_loader = generate_synthetic_data(cfg)
    else:
        print("\n[模式] 使用真实数据")
        train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── 模型构建 ──
    # 暂不支持使用原始数据作为输入，即选择1DCNN作为骨架网络，因为模型定义当中，并没有拆分复数信号的操作
    model = DeepSVDDModel(cfg)
    n_params = count_parameters(model)
    print(f"\n[模型] 骨干: {cfg['model']['backbone']} | "
          f"特征维度: {cfg['model']['feature_dim']} | "
          f"参数量: {n_params:,}")

    # ── 训练 ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device
    )
    trainer.train()

    print("\n训练完成！运行以下命令进行评估：")
    flag = "--synthetic" if args.synthetic else ""
    print(f"  python evaluate.py --config {args.config} {flag}")


if __name__ == "__main__":
    main()
