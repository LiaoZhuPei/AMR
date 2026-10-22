"""
utils.py
公共工具函数

包含:
  - 配置加载
  - 日志设置
  - 随机种子固定
  - 设备选择
  - 合成数据生成（用于无真实数据时的冒烟测试）
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
#  合成数据生成（无真实数据时的替代方案）
# ─────────────────────────────────────────────

def generate_synthetic_data(cfg: dict) -> tuple:
    """
    生成合成的频谱图数据，用于验证整个流程是否工作。

    正类 (MT63/Link11) : 围绕中心 (0.5, 0.5, ...) 的高斯分布
    已知负类           : 围绕不同中心的高斯分布
    未知负类           : 完全不同特征的分布

    Returns:
        train_loader, val_loader, test_loader
    """
    freq_bins = cfg["feature"].get("freq_bins", 129)
    time_frames = cfg["feature"].get("time_frames", 128)
    n_train = 500   # 正类训练样本数

    rng = np.random.default_rng(cfg["data"]["random_seed"])

    def make_batch(n, signal_type="mt63"):
        """生成模拟频谱图"""
        specs = np.zeros((n, 1, freq_bins, time_frames), dtype=np.float32)

        for i in range(n):
            # 基础噪声
            spec = rng.random((freq_bins, time_frames)).astype(np.float32) * 0.1

            if signal_type == "mt63":
                # MT63: 宽带多音，集中在中频段
                f_start = freq_bins // 4
                f_end = 3 * freq_bins // 4
                spec[f_start:f_end, :] += 0.7 + rng.random(
                    (f_end - f_start, time_frames)).astype(np.float32) * 0.2

            elif signal_type == "rtty":
                # RTTY: 窄带两音
                for freq in [freq_bins // 3, freq_bins // 3 + 10]:
                    spec[freq-2:freq+2, :] += 0.8

            elif signal_type == "ssb":
                # SSB: 单边带，低频段
                f_end = freq_bins // 3
                spec[:f_end, :] += 0.6 + rng.random(
                    (f_end, time_frames)).astype(np.float32) * 0.3

            elif signal_type == "am":
                # AM: 载波 + 两侧边带
                fc = freq_bins // 2
                spec[fc-1:fc+1, :] += 0.9
                spec[fc-20:fc-5, :] += 0.5
                spec[fc+5:fc+20, :] += 0.5

            elif signal_type == "psk31":
                # PSK31: 极窄带
                fc = freq_bins // 2 + 15
                spec[fc-1:fc+1, :] += 0.85

            elif signal_type == "morse":
                # Morse: 断续单音
                fc = freq_bins // 3 + 5
                on_off = rng.integers(0, 2, time_frames).astype(np.float32)
                spec[fc-1:fc+1, :] = on_off * 0.9

            elif signal_type == "olivia":
                # Olivia: 多音跳频
                freqs = rng.integers(freq_bins//5, 4*freq_bins//5, 8)
                spec[freqs, :] += 0.7

            spec = np.clip(spec, 0, 1)
            specs[i, 0] = spec

        return specs

    # ── 生成各类数据 ──
    print("[合成数据] 生成训练/验证/测试数据...")

    # 训练集
    x_mt63_tr = make_batch(n_train, "mt63")
    x_rtty_tr = make_batch(200, "rtty")
    x_ssb_tr = make_batch(200, "ssb")
    x_am_tr = make_batch(200, "am")

    x_train = np.concatenate([x_mt63_tr, x_rtty_tr, x_ssb_tr, x_am_tr])
    y_train = np.array([1]*n_train + [0]*200 + [0]*200 + [0]*200)
    cls_train = (["MT63"]*n_train + ["RTTY"]*200 +
                 ["SSB"]*200 + ["AM"]*200)

    # 验证集
    x_mt63_v = make_batch(100, "mt63")
    x_rtty_v = make_batch(50, "rtty")
    x_ssb_v = make_batch(50, "ssb")
    x_am_v = make_batch(50, "am")

    x_val = np.concatenate([x_mt63_v, x_rtty_v, x_ssb_v, x_am_v])
    y_val = np.array([1]*100 + [0]*50 + [0]*50 + [0]*50)
    cls_val = (["MT63"]*100 + ["RTTY"]*50 + ["SSB"]*50 + ["AM"]*50)

    # 测试集（含未知类）
    x_mt63_te = make_batch(100, "mt63")
    x_rtty_te = make_batch(50, "rtty")
    x_ssb_te = make_batch(50, "ssb")
    x_am_te = make_batch(50, "am")
    x_psk31 = make_batch(80, "psk31")
    x_morse = make_batch(80, "morse")
    x_olivia = make_batch(80, "olivia")

    x_test = np.concatenate([
        x_mt63_te, x_rtty_te, x_ssb_te, x_am_te,
        x_psk31, x_morse, x_olivia
    ])
    y_test = np.array([1]*100 + [0]*50 + [0]*50 + [0]*50 +
                      [0]*80 + [0]*80 + [0]*80)
    cls_test = (["MT63"]*100 + ["RTTY"]*50 + ["SSB"]*50 + ["AM"]*50 +
                ["PSK31"]*80 + ["Morse"]*80 + ["Olivia"]*80)

    # ── 打包为 DataLoader ──
    batch_size = cfg["training"]["batch_size"]

    def to_loader(x, y, cls, shuffle=False):
        x_t = torch.from_numpy(x)
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
