"""
utils.py
公共工具函数

包含:
  - 配置加载
  - 日志设置
  - 随机种子固定
  - 设备选择
  - 合成数据生成（用于无真实数据时的冒烟测试，多球体 Deep SVDD 版本）
"""

import os
import random
import logging
import yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(level: int = logging.INFO,
                  log_file: str = None):
    """配置日志"""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )


def set_seed(seed: int = 42):
    """固定所有随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """选择计算设备"""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] 使用 GPU: {torch.cuda.get_device_name(0)}")
    elif prefer_cuda and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] 使用 Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] 使用 CPU")
    return device


# ─────────────────────────────────────────────
#  合成数据生成（无真实数据时的替代方案，多球体版本）
# ─────────────────────────────────────────────

def generate_synthetic_data(cfg: dict) -> tuple:
    """
    生成合成的1D时域信号数据，用于验证多球体 Deep SVDD 整个流程是否工作。

    每个目标类别 (target_classes) : 对应一个特定频率的正弦波 + 噪声，
                                     标签为该类别在列表中的索引 (0..K-1)
    已知负类   (known_negative_classes): 训练/验证阶段可见，波形特征介于
                                     目标类之间，标签统一为 -1
    未知类     (unknown_classes)      : 仅出现在测试集中，波形特征差异更大
                                     （更高频率/更强噪声），标签统一为 -1

    Returns:
        train_loader, val_loader, test_loader
    """
    signal_length = cfg.get("feature", {}).get("signal_length", 1024)
    seed = cfg["data"].get("random_seed", 42)
    batch_size = cfg["training"]["batch_size"]

    target_classes = cfg["data"]["target_classes"]
    known_neg_classes = cfg["data"].get("known_negative_classes", [])
    unknown_classes_cfg = cfg["data"].get("unknown_classes", [])

    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, signal_length, dtype=np.float32)

    def make_wave_batch(n: int, base_freq: float, amp: float = 10000.0,
                        noise_std: float = 2000.0) -> np.ndarray:
        """
        生成 n 条 1D 时域波形: 正弦波 + 高斯噪声。

        Args:
            n:          样本数
            base_freq:  基础频率（决定该"类别"的波形特征）
            amp:        信号幅值
            noise_std:  噪声标准差
        Returns:
            (n, 1, signal_length) float32
        """
        signals = np.zeros((n, 1, signal_length), dtype=np.float32)
        for i in range(n):
            phase = rng.uniform(0, 2 * np.pi)
            wave = np.sin(2 * np.pi * base_freq * t + phase) * amp
            noise = rng.normal(0, noise_std, size=signal_length).astype(np.float32)
            signals[i, 0] = wave + noise
        return signals

    # ── 为每个类别分配一个独立的基础频率，保证类间可分 ──
    # 目标类: 低频段，频率间隔较大，便于形成清晰簇
    n_target = len(target_classes)
    target_freqs = {
        name: (idx + 1) * 5.0 for idx, name in enumerate(target_classes)
    }
    # 已知负类: 频率落在目标类频率之间的间隙，制造"容易混淆但仍需拒识"的负样本
    known_neg_freqs = {
        name: (idx + 1) * 5.0 + 2.5 for idx, name in enumerate(known_neg_classes)
    }
    # 未知类: 更高频段 + 更强噪声，模拟"训练时从未见过的模式"
    unknown_freqs = {
        name: (n_target + idx + 1) * 6.0 for idx, name in enumerate(unknown_classes_cfg)
    }

    print("[合成数据] 生成训练/验证/测试数据...")
    print(f"    目标类({n_target}): {target_classes}")
    print(f"    已知负类({len(known_neg_classes)}): {known_neg_classes}")
    print(f"    未知类({len(unknown_classes_cfg)}, 仅测试): {unknown_classes_cfg}")

    def build_split(n_per_target: int, n_per_known_neg: int,
                    n_per_unknown: int = 0):
        """
        构建一个 split（train/val/test）的信号、标签、类名列表。

        Args:
            n_per_target:    每个目标类别生成的样本数
            n_per_known_neg: 每个已知负类生成的样本数
            n_per_unknown:   每个未知类生成的样本数（仅test使用）
        """
        sig_list, label_list, cls_list = [], [], []

        for idx, name in enumerate(target_classes):
            sig = make_wave_batch(n_per_target, target_freqs[name])
            sig_list.append(sig)
            label_list.extend([idx] * n_per_target)
            cls_list.extend([name] * n_per_target)

        for name in known_neg_classes:
            sig = make_wave_batch(n_per_known_neg, known_neg_freqs[name],
                                  amp=8000.0, noise_std=2500.0)
            sig_list.append(sig)
            label_list.extend([-1] * n_per_known_neg)
            cls_list.extend([name] * n_per_known_neg)

        if n_per_unknown > 0:
            for name in unknown_classes_cfg:
                sig = make_wave_batch(n_per_unknown, unknown_freqs[name],
                                      amp=12000.0, noise_std=3500.0)
                sig_list.append(sig)
                label_list.extend([-1] * n_per_unknown)
                cls_list.extend([name] * n_per_unknown)

        x = np.concatenate(sig_list, axis=0) if sig_list else np.zeros(
            (0, 1, signal_length), dtype=np.float32)
        y = np.array(label_list, dtype=np.int64)
        return x, y, cls_list

    # 训练集：目标类每类200条，已知负类每类200条
    x_train, y_train, cls_train = build_split(
        n_per_target=200, n_per_known_neg=200
    )
    # 验证集：目标类每类50条，已知负类每类50条
    x_val, y_val, cls_val = build_split(
        n_per_target=50, n_per_known_neg=50
    )
    # 测试集：目标类每类60条，已知负类每类60条，未知类每类60条
    x_test, y_test, cls_test = build_split(
        n_per_target=60, n_per_known_neg=60, n_per_unknown=60
    )

    def to_loader(x, y, cls, shuffle=False):
        # 简单 z-score 归一化，与真实数据 pipeline 保持一致
        mean = x.mean(axis=2, keepdims=True)
        std = x.std(axis=2, keepdims=True) + 1e-8
        x = (x - mean) / std

        x_t = torch.from_numpy(x.astype(np.float32))
        y_t = torch.from_numpy(y).long()
        ds = _LabeledDataset(x_t, y_t, cls)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    train_loader = to_loader(x_train, y_train, cls_train, shuffle=True)
    val_loader = to_loader(x_val, y_val, cls_val)
    test_loader = to_loader(x_test, y_test, cls_test)

    print(f"[合成数据] Train: {len(x_train)}, "
          f"Val: {len(x_val)}, Test: {len(x_test)}")

    return train_loader, val_loader, test_loader


class _LabeledDataset(torch.utils.data.Dataset):
    """带类名的数据集包装器"""

    def __init__(self, x, y, cls_names):
        self.x = x
        self.y = y
        self.cls_names = cls_names

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.cls_names[idx]


def count_parameters(model) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)