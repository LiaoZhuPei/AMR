"""
dataset.py
数据加载与预处理模块

数据集: Panoradio HF  (panoradio-sdr.de)
  dataset_panoradio_hf.npy
    shape = (172800, 2048)  dtype = complex64
    每行 = 1 个信号样本，2048 个复数 IQ 点，fs=6000 Hz，时长≈341ms
    功率归一化为 1，频偏 ±250 Hz，SNR: 25/20/15/10/5/0/-5/-10 dB

  dataset_panoradio_hf_tags.csv
    idx | mode          | snr
    0   | morse         | 25
    ...
    9714| psk31         | 25

18 种 mode（数据集原始名）:
  morse, psk31, psk63, qpsk31,
  rtty45_170, rtty50_170, rtty100_850,
  olivia8_250, olivia16_500, olivia16_1000, olivia32_1000,
  dominoex11, mt63_1000, navtex,
  usb, lsb, am, fax

config.yaml 中的简称 → 匹配规则（前缀匹配 mode 字段）:
  "mt63_1000"  → target_class: "mt63_1000"
  "RTTY"       → 匹配 rtty45_170 / rtty50_170 / rtty100_850
  "SSB"        → 匹配 usb / lsb
  "AM"         → 匹配 am
  "PSK31"      → 匹配 psk31
  "Morse"      → 匹配 morse
  "Olivia"     → 匹配 olivia8_250 / olivia16_500 / …

输出张量: (1, n_fft, time_frames)  float32 频谱图，单通道，供 CNN2D 使用
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
#  标签定义
# ──────────────────────────────────────────────────────────────────
LABEL_POSITIVE = 1   # 正类  (mt63_1000 → 模拟 Link11)
LABEL_NEGATIVE = 0   # 负类  (已知 或 未知)


# ──────────────────────────────────────────────────────────────────
#  类名匹配：config 简称 → tags.csv 原始 mode 字符串
# ──────────────────────────────────────────────────────────────────
# config 写什么 → 用什么前缀去匹配 tags.csv 的 mode 列
_ALIAS_PREFIX: dict[str, list[str]] = {
    # 完全匹配
    "mt63_1000":  ["mt63_1000"],
    # 常见简称（不区分大小写前缀）
    "rtty":       ["rtty"],          # rtty45_170, rtty50_170, rtty100_850
    "ssb":        ["usb", "lsb"],    # Single-Sideband upper/lower
    "am":         ["am"],
    "psk31":      ["psk31"],
    "morse":      ["morse"],
    "olivia":     ["olivia"],        # olivia8_250, olivia16_500, …
    # 其他类（通常不用，但保留以便扩展）
    "psk63":      ["psk63"],
    "qpsk31":     ["qpsk31"],
    "dominoex11": ["dominoex11"],
    "navtex":     ["navtex"],
    "fax":        ["fax"],
}


def _get_prefixes(alias: str) -> list[str]:
    """返回 alias 对应的前缀列表（不区分大小写）"""
    key = alias.lower()
    return _ALIAS_PREFIX.get(key, [key])


def _mode_matches(mode: str, alias: str) -> bool:
    """
    判断 tags.csv 中的 mode 值是否属于 config 里的 alias。

    例:
      "rtty45_170" 匹配 alias="RTTY"   → True
      "usb"        匹配 alias="SSB"    → True
      "am"         匹配 alias="am"     → True
      "psk63"      匹配 alias="PSK31"  → False
    """
    mode_lower = mode.lower().strip()
    for prefix in _get_prefixes(alias):
        if mode_lower.startswith(prefix.lower()):
            return True
    return False


# ──────────────────────────────────────────────────────────────────
#  .npy + tags.csv 加载（全局缓存，多次实例化只读一次磁盘）
# ──────────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _load_dataset(npy_path: str, tags_path: str) -> tuple:
    """
    加载数据文件（带内存缓存）。

    Returns:
        data : np.ndarray, shape=(172800, 2048), dtype=complex64
        df   : pd.DataFrame, columns=[idx, mode, snr]
    """
    if npy_path in _CACHE:
        return _CACHE[npy_path]

    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"\n[数据缺失] 找不到: {npy_path}\n"
            "请从 https://panoradio-sdr.de/radio-signal-classification-dataset/\n"
            "下载 dataset_panoradio_hf.npy 并放入 data/raw/\n"
            "（无数据时可用 python smoke_test.py 用合成数据验证流程）"
        )
    if not os.path.exists(tags_path):
        raise FileNotFoundError(
            f"\n[数据缺失] 找不到: {tags_path}\n"
            "请下载 dataset_panoradio_hf_tags.csv 并放入 data/raw/"
        )

    logger.info(f"正在加载: {npy_path}  (首次约需 10~30s)")
    data = np.load(npy_path)          # (172800, 2048) complex64
    logger.info(f"已加载: shape={data.shape}, dtype={data.dtype}")

    df = pd.read_csv(tags_path, sep=r"[|,;\t]", engine="python",
                     skipinitialspace=True)
    # 规范列名：去掉空格，统一小写
    df.columns = [c.strip().lower() for c in df.columns]
    # 期望列: idx / mode / snr
    if "mode" not in df.columns:
        raise ValueError(f"tags.csv 中未找到 'mode' 列，实际列: {list(df.columns)}")

    df["mode"] = df["mode"].str.strip().str.lower()

    assert len(df) == len(data), (
        f"标签行数({len(df)}) 与数据行数({len(data)}) 不匹配"
    )

    logger.info(f"tags.csv 已加载: {len(df)} 行 | "
                f"mode 类别: {sorted(df['mode'].unique())}")

    _CACHE[npy_path] = (data, df)
    return data, df


# ──────────────────────────────────────────────────────────────────
#  IQ 复数信号 → STFT 幅度频谱图
# ──────────────────────────────────────────────────────────────────

def iq_to_spectrogram(iq: np.ndarray,
                      n_fft: int = 256,
                      hop_length: int = 64,
                      normalize: bool = True) -> np.ndarray:
    """
    复数 IQ 信号 → STFT 双边幅度频谱图（dB, fftshift 令0Hz居中）。

    Args:
        iq         : shape=(2048,), dtype=complex64
        n_fft      : FFT 窗长
        hop_length : 帧移（点数）
        normalize  : 是否归一化到 [0,1]

    Returns:
        spec : shape=(n_fft, time_frames), float32
               对复数信号做双边 STFT，保留负频率，
               有助于区分 USB/LSB 等依赖频谱不对称的信号。
    """
    # 参见注解1
    _, _, Zxx = stft(
        iq,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=False   # 双边：保留负频率
    )
    # Zxx: (n_fft, time_frames), complex
    spec = np.abs(Zxx).astype(np.float32)
    spec = np.fft.fftshift(spec, axes=0)   # 0Hz 移到中心行

    # 转 dB，压缩动态范围
    spec = 20.0 * np.log10(spec + 1e-8)

    if normalize:
        lo, hi = spec.min(), spec.max()
        spec = (spec - lo) / (hi - lo + 1e-8)

    return spec   # (n_fft, time_frames)


# ──────────────────────────────────────────────────────────────────
#  主数据集类
# ──────────────────────────────────────────────────────────────────

class PanoradioDataset(Dataset):
    """
    Panoradio HF 开放集信号数据集。

    Split 策略（在样本级别随机划分，固定种子保证无泄漏）:
      train : train_ratio (默认 70%) 的样本
      val   : val_ratio   (默认 15%) 的样本
      test  : 剩余        (默认 15%) 的样本

    训练/验证阶段: 仅加载正类 + 已知负类
    测试阶段:      加载所有类（含未知负类，检验开放集拒识能力）

    __getitem__ 返回:
      x        : Tensor (1, n_fft, time_frames) float32 — 频谱图
      label    : Tensor scalar int64  — 1=Link11, 0=非Link11
      cls_name : str                  — config 中的类别简称
    """

    def __init__(self,
                 cfg: dict,
                 mode: str = "train",
                 transform=None):
        """
        Args:
            cfg       : 完整配置字典（来自 config.yaml）
            mode      : "train" | "val" | "test"
            transform : 可选的张量级数据增强，callable(tensor)->tensor
        """
        assert mode in ("train", "val", "test"), \
            f"mode 必须是 train/val/test，得到: {mode}"

        self.cfg       = cfg
        self.mode      = mode
        self.transform = transform

        data_cfg = cfg["data"]
        feat_cfg = cfg["feature"]

        self.n_fft      = feat_cfg["n_fft"]
        self.hop_length = feat_cfg["hop_length"]
        self.normalize  = feat_cfg["normalize"]

        self.target_class      = data_cfg["target_class"]      # "mt63_1000"
        self.known_negatives   = data_cfg["known_negatives"]   # ["rtty","ssb","am"]
        self.unknown_negatives = data_cfg["unknown_negatives"] # ["psk31","morse","olivia"]

        # 加载原始数组 + 标签 DataFrame（带全局缓存）
        raw_dir    = data_cfg["raw_dir"]
        npy_path   = os.path.join(raw_dir, "dataset_panoradio_hf.npy")
        tags_path  = os.path.join(raw_dir, "dataset_panoradio_hf_tags.csv")
        self._data, self._df = _load_dataset(npy_path, tags_path)

        # 构建本 split 的 (行索引, label, cls_name) 列表
        self._index: list[tuple[int, int, str]] = []
        self._build_index(data_cfg, mode)

        n_pos = sum(1 for _, l, _ in self._index if l == LABEL_POSITIVE)
        n_neg = len(self._index) - n_pos
        logger.info(
            f"[{mode.upper():5s}] {len(self._index):6d} 样本 "
            f"| 正类(Link11): {n_pos}  负类: {n_neg}"
        )

    # ── 索引构建 ────────────────────────────────────────────────────

    def _build_index(self, data_cfg: dict, mode: str):
        """
        按类别 + split 比例筛选行索引，写入 self._index。
        在样本层面随机打乱后按比例切分，确保同一个信号不会
        同时出现在训练集和测试集。
        """
        rng = np.random.default_rng(data_cfg["random_seed"])
        train_r = float(data_cfg["train_ratio"])
        val_r   = float(data_cfg["val_ratio"])

        # 本 split 需要的类别列表
        if mode in ("train", "val"):
            aliases = [self.target_class] + self.known_negatives
        else:
            aliases = (
                [self.target_class]
                + self.known_negatives
                + self.unknown_negatives
            )

        for alias in aliases:
            # 找所有匹配行
            mask = self._df["mode"].apply(lambda m: _mode_matches(m, alias))
            rows = self._df.index[mask].to_numpy()

            if len(rows) == 0:
                logger.warning(
                    f"  ⚠ 类别 '{alias}' 在 tags.csv 中未匹配到任何行！\n"
                    f"    实际 mode 列表: {sorted(self._df['mode'].unique())}\n"
                    f"    请检查 config.yaml 中的类名或 _ALIAS_PREFIX 映射。"
                )
                continue

            # 固定种子打乱，切分
            shuffled = rng.permutation(rows)
            n       = len(shuffled)
            n_train = int(n * train_r)
            n_val   = int(n * val_r)

            if mode == "train":
                selected = shuffled[:n_train]
            elif mode == "val":
                selected = shuffled[n_train: n_train + n_val]
            else:   # test
                selected = shuffled[n_train + n_val:]
                if len(selected) == 0:
                    selected = shuffled   # 极少样本时降级为全量

            label = LABEL_POSITIVE if alias == self.target_class else LABEL_NEGATIVE
            for row_idx in selected:
                self._index.append((int(row_idx), label, alias))

    # ── Dataset 接口 ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        row_idx, label, cls_name = self._index[idx]

        # shape=(2048,) complex64 → 直接取，不需要分段
        iq = self._data[row_idx]

        # IQ → 频谱图 (n_fft, time_frames)
        spec = iq_to_spectrogram(
            iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalize=self.normalize,
        )

        # 增加通道维: (1, n_fft, time_frames)
        x = torch.from_numpy(spec).unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        return x, torch.tensor(label, dtype=torch.long), cls_name


# ──────────────────────────────────────────────────────────────────
#  DataLoader 工厂（供 train.py / evaluate.py 调用）
# ──────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict):
    """
    一次性构建 train / val / test 三个 DataLoader。

    数据加载说明:
      - .npy 文件只读一次（全局缓存），三个 Dataset 共享内存
      - num_workers=2 异步预取；persistent_workers 避免每 epoch 重启进程

    Returns:
        train_loader, val_loader, test_loader
    """
    bs = cfg["training"]["batch_size"]

    train_ds = PanoradioDataset(cfg, mode="train")
    val_ds   = PanoradioDataset(cfg, mode="val")
    test_ds  = PanoradioDataset(cfg, mode="test")

    def _make(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        )

    return _make(train_ds, True), _make(val_ds, False), _make(test_ds, False)


# ──────────────────────────────────────────────────────────────────
#  独立运行: 格式验证 + 快速自检
#  用法: python src/dataset.py
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg_path = "configs/config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir   = cfg["data"]["raw_dir"]
    npy_path  = os.path.join(raw_dir, "dataset_panoradio_hf.npy")
    tags_path = os.path.join(raw_dir, "dataset_panoradio_hf_tags.csv")

    sep = "─" * 55
    print(f"\n{'='*55}")
    print("  dataset.py 自检")
    print(f"{'='*55}")

    # 1. 文件检查
    print(f"\n[1] 文件检查")
    all_ok = True
    for p in (npy_path, tags_path):
        ex = os.path.exists(p)
        sz = f"{os.path.getsize(p)/1e9:.2f} GB" if ex else "未找到"
        print(f"  {'✓' if ex else '✗'}  {p}  ({sz})")
        if not ex:
            all_ok = False
    if not all_ok:
        print("\n  请先下载数据集。无数据时可运行: python smoke_test.py")
        sys.exit(0)

    # 2. 加载
    print(f"\n[2] 加载数据")
    data, df = _load_dataset(npy_path, tags_path)
    print(f"  npy  shape : {data.shape}")
    print(f"  npy  dtype : {data.dtype}")
    print(f"  tags 行数  : {len(df)}")
    print(f"  tags mode 列表 : {sorted(df['mode'].unique())}")
    print(f"  tags snr  列表 : {sorted(df['snr'].unique()) if 'snr' in df.columns else '无'}")

    # 3. 单样本
    print(f"\n[3] 单样本处理")
    sample = data[0]
    print(f"  IQ  : shape={sample.shape}, dtype={sample.dtype}, "
          f"is_complex={np.iscomplexobj(sample)}")
    spec = iq_to_spectrogram(sample,
                             n_fft=cfg["feature"]["n_fft"],
                             hop_length=cfg["feature"]["hop_length"])
    print(f"  频谱: shape={spec.shape}, range=[{spec.min():.3f}, {spec.max():.3f}]")

    # 4. 类名匹配验证
    print(f"\n[4] 类名匹配验证")
    all_modes = sorted(df["mode"].unique())
    aliases = (
        [cfg["data"]["target_class"]]
        + cfg["data"]["known_negatives"]
        + cfg["data"]["unknown_negatives"]
    )
    for alias in aliases:
        matched = [m for m in all_modes if _mode_matches(m, alias)]
        ok = "✓" if matched else "✗ 未匹配！"
        print(f"  {ok}  '{alias}' → {matched}")

    # 5. Dataset
    print(f"\n[5] Dataset 构建")
    ds = PanoradioDataset(cfg, mode="train")
    x, y, cls = ds[0]
    print(f"  训练集大小 : {len(ds)}")
    print(f"  x.shape    : {tuple(x.shape)}")
    print(f"  y / cls    : {y.item()} / {cls}")

    print(f"\n  自检完成 ✓\n")
