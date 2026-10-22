import argparse
import logging
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, setup_logging, set_seed, get_device
from src.utils import generate_synthetic_data, count_parameters
from src.model import MultiSphereDeepSVDDModel
from src.trainer import Trainer
from src.dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(
        description="多目标开放集识别 - 训练脚本 (Multi-Sphere Deep SVDD)"
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


def _validate_cfg(cfg: dict):
    """
    多球体模型关键配置一致性检查，避免因配置疏漏导致隐蔽错误。
    """
    num_classes = cfg["model"]["num_classes"]
    target_classes = cfg["data"]["target_classes"]

    if num_classes != len(target_classes):
        raise ValueError(
            f"[配置错误] model.num_classes({num_classes}) 与 "
            f"data.target_classes 长度({len(target_classes)}) 不一致！\n"
            f"target_classes = {target_classes}"
        )

    overlap = set(target_classes) & set(cfg["data"].get("known_negative_classes", []))
    if overlap:
        raise ValueError(
            f"[配置错误] target_classes 与 known_negative_classes 存在重叠类别: {overlap}"
        )

    overlap2 = set(target_classes) & set(cfg["data"].get("unknown_classes", []))
    if overlap2:
        raise ValueError(
            f"[配置错误] target_classes 与 unknown_classes 存在重叠类别: {overlap2}"
        )


def main():
    args = parse_args()

    # ── 初始化 ──
    setup_logging(level=logging.INFO,
                  log_file="results/train.log")
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    _validate_cfg(cfg)
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
    model = MultiSphereDeepSVDDModel(cfg)
    n_params = count_parameters(model)
    print(f"\n[模型] 骨干: {cfg['model']['backbone']} | "
          f"特征维度: {cfg['model']['feature_dim']} | "
          f"已知类别数(K): {model.num_classes} | "
          f"参数量: {n_params:,}")
    print(f"[类别] 目标类: {cfg['data']['target_classes']}")
    if cfg["data"].get("known_negative_classes"):
        print(f"[类别] 已知负类: {cfg['data']['known_negative_classes']}")
    if cfg["data"].get("unknown_classes"):
        print(f"[类别] 未知类(仅测试): {cfg['data']['unknown_classes']}")

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