"""
dataset.py
数据加载与预处理模块

数据集: Panoradio HF
  - 信号文件: dataset_panoradio_hf.npy
      shape = (172800, 2048), dtype=complex64
      每行: 2048个IQ采样点, fs=6000Hz, 功率归一化到1
      频偏: ±250Hz, SNR: 25/20/15/10/5/0/-5/-10 dB
      信道: CCIR 520 fading
  - 标签文件: dataset_panoradio_hf_tags.csv
      列: idx, mode, snr
  - 18种调制模式:
      morse, psk31, psk63, qpsk31,
      rtty45_170, rtty50_170, rtty100_850,
      olivia8_250, olivia16_500, olivia16_1000, olivia32_1000,
      dominoex11, mt63_1000, navtex,
      usb, lsb, am, fax

开放集设置:
  - 正类  (target)  : mt63_1000
  - 已知负类        : 训练时可见的其他调制
  - 未知负类        : 仅测试时出现的调制
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  常量
# ─────────────────────────────────────────────

# Panoradio HF 数据集采样率
FS = 6000  # Hz

# 数据集全部模式（共18种）
ALL_MODES = [
    "morse", "psk31", "psk63", "qpsk31",
    "rtty45_170", "rtty50_170", "rtty100_850",
    "olivia8_250", "olivia16_500", "olivia16_1000", "olivia32_1000",
    "dominoex11", "mt63_1000", "navtex",
    "usb", "lsb", "am", "fax"
]

# 开放集标签
LABEL_POSITIVE = 1   # 目标正类
LABEL_NEGATIVE = 0   # 负类（已知或未知）


# ─────────────────────────────────────────────
#  数据集加载
# ─────────────────────────────────────────────

def load_panoradio_dataset(npy_path: str,
                            csv_path: str
                            ) -> tuple[np.ndarray, pd.DataFrame]:
    """
    加载 Panoradio HF 数据集的信号与标签。

    Args:
        npy_path: .npy 文件路径，信号 shape=(172800, 2048), complex
        csv_path: .csv 标签文件路径，列为 idx/mode/snr

    Returns:
        signals : np.ndarray, shape=(N, 2048), complex64
        tags    : pd.DataFrame, 列=[idx, mode, snr], 行数=N
    """
    logger.info(f"Loading signals from: {npy_path}")
    signals = np.load(npy_path)                    # (N, 2048)

    if signals.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {signals.shape}")

    # 确保复数类型
    if not np.iscomplexobj(signals):
        logger.warning(
            "Signal array is not complex dtype. "
            "Treating as real (imaginary=0)."
        )
        signals = signals.astype(np.complex64)
    else:
        signals = signals.astype(np.complex64)

    logger.info(f"Signals loaded: shape={signals.shape}, dtype={signals.dtype}")

    logger.info(f"Loading tags from: {csv_path}")
    tags = pd.read_csv(csv_path, index_col=0)      # 第0列为 idx

    # 统一列名（兼容有无空格的情况）
    tags.columns = tags.columns.str.strip()

    required_cols = {"mode", "snr"}
    if not required_cols.issubset(tags.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, "
            f"got {list(tags.columns)}"
        )

    if len(signals) != len(tags):
        raise ValueError(
            f"Signal count {len(signals)} != tag count {len(tags)}"
        )

    logger.info(
        f"Tags loaded: {len(tags)} rows, "
        f"modes={sorted(tags['mode'].unique())}"
    )

    return signals, tags


# ─────────────────────────────────────────────
#  IQ → 特征
# ─────────────────────────────────────────────

def iq_to_spectrogram(iq: np.ndarray,
                       n_fft: int = 256,
                       hop_length: int = 64,
                       normalize: bool = True) -> np.ndarray:
    """
    将单条复数 IQ 信号转换为 STFT 幅度频谱图（dB 尺度）。

    IQ 信号直接用于复数 STFT，频率轴覆盖 [-fs/2, +fs/2]。

    Args:
        iq         : 复数 IQ 信号, shape=(L,), complex64
        n_fft      : FFT 点数（窗长）
        hop_length : 帧移
        normalize  : 是否将频谱图归一化到 [0, 1]

    Returns:
        spec: shape=(n_fft, time_frames), float32
              频率轴已 fftshift，低频居中
    """
    # scipy.signal.stft 对复数输入直接支持
    # return_onesided=False → 输出双边谱，频率轴长度 = n_fft
    _, _, Zxx = stft(
        iq,
        fs=FS,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=False,
        boundary=None,
        padded=False
    )
    # Zxx: shape=(n_fft, time_frames)

    # fftshift：将零频移到中央（低频居中显示）
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # 幅度谱 → dB
    spec = np.abs(Zxx).astype(np.float32)
    spec = 20.0 * np.log10(spec + 1e-8)

    if normalize:
        s_min, s_max = spec.min(), spec.max()
        if s_max > s_min:
            spec = (spec - s_min) / (s_max - s_min)
        else:
            spec = np.zeros_like(spec)

    return spec   # (n_fft, time_frames)


def iq_to_amplitude_phase(iq: np.ndarray) -> np.ndarray:
    """
    将 IQ 信号转为 [幅度, 相位差分] 双通道特征（时域）。

    常用于轻量级时域网络（CNN-1D / LSTM）。

    Args:
        iq: shape=(L,), complex64

    Returns:
        feat: shape=(2, L), float32
              feat[0] = 归一化幅度 ∈ [0,1]
              feat[1] = 相邻相位差分（弧度），归一化到 [-1,1]
    """
    amp   = np.abs(iq).astype(np.float32)          # (L,)
    phase = np.angle(iq).astype(np.float32)         # (L,) ∈ [-π, π]
    dphase = np.diff(phase, prepend=phase[0])        # 相位差分

    # 归一化幅度
    max_amp = amp.max()
    if max_amp > 0:
        amp = amp / max_amp

    # 相位差分归一化到 [-1, 1]
    dphase = dphase / np.pi

    return np.stack([amp, dphase], axis=0)           # (2, L)


# ─────────────────────────────────────────────
#  数据集类
# ─────────────────────────────────────────────

class PanoradioDataset(Dataset):
    """
    Panoradio HF 开放集信号识别数据集。

    特性:
      - 支持按 mode / snr 过滤
      - 支持开放集划分（已知类训练，未知类仅测试）
      - 支持两种特征：频谱图 (spectrogram) 或 时域双通道 (iq_2ch)
      - 按信号索引做 train/val/test split（无数据泄漏）

    用法:
        train_ds = PanoradioDataset(signals, tags, cfg, mode="train")
        val_ds   = PanoradioDataset(signals, tags, cfg, mode="val")
        test_ds  = PanoradioDataset(signals, tags, cfg, mode="test")
    """

    def __init__(self,
                 signals : np.ndarray,
                 tags    : pd.DataFrame,
                 cfg     : dict,
                 mode    : str = "train",
                 transform = None):
        """
        Args:
            signals  : shape=(N, 2048), complex64，全量信号
            tags     : DataFrame, 列=[mode, snr]，行数=N
            cfg      : 配置字典（见 configs/config.yaml）
            mode     : "train" | "val" | "test"
            transform: 可选增强，作用于输出 tensor
        """
        super().__init__()
        self.cfg       = cfg
        self.mode      = mode
        self.transform = transform

        data_cfg = cfg["data"]
        feat_cfg = cfg["feature"]

        # ── 配置读取 ──────────────────────────────────────────────
        self.target_class      = data_cfg["target_class"]        # str
        self.known_negatives   = data_cfg["known_negatives"]     # list[str]
        self.unknown_negatives = data_cfg["unknown_negatives"]   # list[str]
        self.snr_filter        = data_cfg.get("snr_filter", None)
            # None → 全 SNR；或 list[int]，如 [0, 5, 10]

        self.feature_type = feat_cfg.get("type", "spectrogram")
            # "spectrogram" | "iq_2ch"
        self.n_fft      = feat_cfg["n_fft"]
        self.hop_length = feat_cfg["hop_length"]
        self.normalize  = feat_cfg["normalize"]

        train_ratio = data_cfg["train_ratio"]
        val_ratio   = data_cfg["val_ratio"]
        seed        = data_cfg["random_seed"]

        # ── 确定本 split 使用的类 ─────────────────────────────────
        if mode in ("train", "val"):
            active_modes = [self.target_class] + self.known_negatives
        else:  # test：已知类 + 未知类
            active_modes = (
                [self.target_class] +
                self.known_negatives +
                self.unknown_negatives
            )

        # ── 过滤：仅保留 active_modes 的行 ───────────────────────
        mode_mask = tags["mode"].isin(active_modes)
        if self.snr_filter is not None:
            snr_mask  = tags["snr"].isin(self.snr_filter)
            mask      = mode_mask & snr_mask
        else:
            mask      = mode_mask

        filtered_idx = np.where(mask.values)[0]   # 全局行索引

        # ── train/val/test 按全局索引划分（类内比例均衡） ─────────
        # 为保证每种 mode×snr 组合在各 split 中均有覆盖，
        # 分层 shuffle：按 (mode, snr) 分组后各自划分
        rng = np.random.default_rng(seed)
        split_indices = self._stratified_split(
            filtered_idx, tags, rng,
            train_ratio, val_ratio, mode
        )

        # ── 构建样本列表 ──────────────────────────────────────────
        # 存储 (全局信号索引, label, mode_name) 而非预计算特征，
        # 节省内存；特征在 __getitem__ 中实时计算。
        # 若数据集较小可改为预计算。
        self._signals = signals          # 共享引用，不复制
        self._tags    = tags

        self.index_list = []             # list of (global_idx, label, mode_name)
        for gidx in split_indices:
            mode_name = tags.iloc[gidx]["mode"]
            label     = (LABEL_POSITIVE
                         if mode_name == self.target_class
                         else LABEL_NEGATIVE)
            self.index_list.append((int(gidx), label, mode_name))

        pos = sum(1 for _, l, _ in self.index_list if l == LABEL_POSITIVE)
        neg = len(self.index_list) - pos
        logger.info(
            f"[{mode}] total={len(self.index_list)}, "
            f"positive={pos}, negative={neg}, "
            f"active_modes={active_modes}"
        )

    # ──────────────────────────────────────────────────────────────
    #  分层划分
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _stratified_split(filtered_idx : np.ndarray,
                           tags         : pd.DataFrame,
                           rng          : np.random.Generator,
                           train_ratio  : float,
                           val_ratio    : float,
                           mode         : str) -> np.ndarray:
        """
        按 (mode, snr) 分层划分，确保每个子组的样本
        均匀分布在 train/val/test 三个 split。

        Returns:
            本 split 的全局行索引数组
        """
        train_idx, val_idx, test_idx = [], [], []

        # 按 (mode, snr) 分组
        group_keys = tags.iloc[filtered_idx][["mode", "snr"]]
        group_keys = group_keys.copy()
        group_keys["_gidx"] = filtered_idx   # 挂载全局索引

        for _, grp in group_keys.groupby(["mode", "snr"]):
            gidxs = grp["_gidx"].values.copy()
            rng.shuffle(gidxs)
            n       = len(gidxs)
            n_train = max(1, int(n * train_ratio))
            n_val   = max(1, int(n * val_ratio))

            train_idx.extend(gidxs[:n_train])
            val_idx.extend  (gidxs[n_train: n_train + n_val])
            test_idx.extend (gidxs[n_train + n_val:])

        mapping = {
            "train": np.array(train_idx, dtype=np.int64),
            "val"  : np.array(val_idx,   dtype=np.int64),
            "test" : np.array(test_idx,  dtype=np.int64),
        }
        return mapping[mode]

    # ──────────────────────────────────────────────────────────────
    #  Dataset 接口
    # ──────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        gidx, label, mode_name = self.index_list[idx]

        iq = self._signals[gidx]          # (2048,) complex64

        # ── 特征提取 ──────────────────────────────────────────────
        if self.feature_type == "spectrogram":
            feat = iq_to_spectrogram(
                iq,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                normalize=self.normalize
            )                              # (n_fft, time_frames)
            x = torch.from_numpy(feat).unsqueeze(0)
            # → (1, n_fft, time_frames)

        elif self.feature_type == "iq_2ch":
            feat = iq_to_amplitude_phase(iq)   # (2, 2048)
            x = torch.from_numpy(feat)

        else:
            raise ValueError(
                f"Unknown feature type: {self.feature_type}. "
                f"Choose 'spectrogram' or 'iq_2ch'."
            )

        if self.transform:
            x = self.transform(x)

        return (
            x,
            torch.tensor(label, dtype=torch.long),
            mode_name
        )


# ─────────────────────────────────────────────
#  DataLoader 工厂函数
# ─────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    """
    一次性加载数据集，构建 train/val/test DataLoader。

    数据集文件路径从 cfg["data"] 读取:
        npy_path: dataset_panoradio_hf.npy 路径
        csv_path: dataset_panoradio_hf_tags.csv 路径

    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg   = cfg["data"]
    npy_path   = data_cfg["npy_path"]
    csv_path   = data_cfg["csv_path"]
    batch_size = cfg["training"]["batch_size"]
    num_workers= cfg["training"].get("num_workers", 4)

    # 一次 IO，三个 split 共享内存
    signals, tags = load_panoradio_dataset(npy_path, csv_path)

    train_ds = PanoradioDataset(signals, tags, cfg, mode="train")
    val_ds   = PanoradioDataset(signals, tags, cfg, mode="val")
    test_ds  = PanoradioDataset(signals, tags, cfg, mode="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  冒烟测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s"
    )

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("  冒烟测试：使用合成数据验证 PanoradioDataset")
    print("=" * 60)

    # ── 合成信号与标签 ────────────────────────────────────────────
    N_FAKE  = 1800    # 每种 mode×snr 组合 10 条
    N_SAMPS = 2048
    rng_np  = np.random.default_rng(42)

    # 随机复数 IQ（功率归一化）
    fake_iq  = (rng_np.standard_normal((N_FAKE, N_SAMPS)) +
                1j * rng_np.standard_normal((N_FAKE, N_SAMPS)))
    fake_iq /= np.sqrt((np.abs(fake_iq)**2).mean(axis=1, keepdims=True))
    fake_iq  = fake_iq.astype(np.complex64)

    # 构造标签 DataFrame（18 mode × 10 snr × 10 条 = 1800）
    SNRS  = [25, 20, 15, 10, 5, 0, -5, -10]
    rows  = []
    idx   = 0
    for mode_name in ALL_MODES:
        for snr_val in SNRS[:10]:             # 取前10个SNR做演示
            for _ in range(10):
                rows.append({"idx": idx, "mode": mode_name, "snr": snr_val})
                idx += 1
                if idx >= N_FAKE:
                    break
            if idx >= N_FAKE:
                break
        if idx >= N_FAKE:
            break

    fake_tags = pd.DataFrame(rows).set_index("idx")
    fake_tags.index.name = None   # 与真实 CSV 加载后一致

    # ── 测试 iq_to_spectrogram ────────────────────────────────────
    print("\n[1] iq_to_spectrogram")
    spec = iq_to_spectrogram(
        fake_iq[0],
        n_fft=cfg["feature"]["n_fft"],
        hop_length=cfg["feature"]["hop_length"]
    )
    print(f"    spec shape : {spec.shape}")
    print(f"    value range: [{spec.min():.3f}, {spec.max():.3f}]")

    # ── 测试 iq_to_amplitude_phase ────────────────────────────────
    print("\n[2] iq_to_amplitude_phase")
    ap = iq_to_amplitude_phase(fake_iq[0])
    print(f"    feat shape : {ap.shape}")
    print(f"    amp  range : [{ap[0].min():.3f}, {ap[0].max():.3f}]")
    print(f"    dphase range: [{ap[1].min():.3f}, {ap[1].max():.3f}]")

    # ── 测试 PanoradioDataset ─────────────────────────────────────
    print("\n[3] PanoradioDataset (spectrogram mode)")
    train_ds = PanoradioDataset(fake_iq, fake_tags, cfg, mode="train")
    val_ds   = PanoradioDataset(fake_iq, fake_tags, cfg, mode="val")
    test_ds  = PanoradioDataset(fake_iq, fake_tags, cfg, mode="test")

    print(f"    train size : {len(train_ds)}")
    print(f"    val   size : {len(val_ds)}")
    print(f"    test  size : {len(test_ds)}")

    x, y, name = train_ds[0]
    print(f"\n    sample x shape : {x.shape}")
    print(f"    sample label   : {y.item()} ({name})")

    # ── DataLoader 迭代一批 ───────────────────────────────────────
    print("\n[4] DataLoader batch test")
    loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    xb, yb, nb = next(iter(loader))
    print(f"    batch x : {xb.shape}")
    print(f"    batch y : {yb.shape}, unique={yb.unique().tolist()}")
    print(f"    batch names sample: {list(nb[:4])}")

    print("\n" + "=" * 60)
    print("  dataset.py 所有测试通过 ✓")
    print("=" * 60)