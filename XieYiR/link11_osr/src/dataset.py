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
LABEL_POSITIVE = 1   # 正类 (目标类)
LABEL_NEGATIVE = 0   # 负类 (已知负类 或 未知负类)

# ──────────────────────────────────────────────────────────────────
#  类名匹配：config 简称 → datalabel_rtl.csv 原始 mode 字符串
# ──────────────────────────────────────────────────────────────────
_ALIAS_PREFIX: dict[str, list[str]] = {
    "aprs":                ["aprs"],
    "fm_broadcast":        ["fm_broadcast"], 
    "frs_gmrs":            ["frs_gmrs"], 
    "ism_sensors":         ["ism_sensors"],
    "noaa_weather":        ["noaa_weather"],
    "noise":               ["noise"],
    "pager":               ["pager"], 
}


def _get_prefixes(alias: str) -> list[str]:
    """返回 alias 对应的前缀列表（不区分大小写）"""
    key = alias.lower().strip()
    return _ALIAS_PREFIX.get(key, [key])


def _mode_matches(mode: str, alias: str) -> bool:
    """判断 CSV 中的 mode 值是否属于 config 里的 alias"""
    mode_lower = mode.lower().strip()
    for prefix in _get_prefixes(alias):
        if mode_lower.startswith(prefix.lower()):
            return True
    return False


# ──────────────────────────────────────────────────────────────────
#  .npy + tags.csv 加载（全局缓存）
# ──────────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _load_dataset(npy_path: str, tags_path: str) -> tuple:
    """加载数据文件（全局内存缓存，避免重复加载）"""
    if npy_path in _CACHE:
        return _CACHE[npy_path]

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"\n[数据缺失] 找不到: {npy_path}")
    if not os.path.exists(tags_path):
        raise FileNotFoundError(f"\n[数据缺失] 找不到: {tags_path}")

    logger.info(f"正在加载数据: {npy_path} ...")
    data = np.load(npy_path)
    logger.info(f"数据加载完成: shape={data.shape}, dtype={data.dtype}")

    df = pd.read_csv(tags_path, sep=r"[|,;\t]", engine="python", skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if "mode" not in df.columns:
        raise ValueError(f"CSV 中未找到 'mode' 列，实际列名: {list(df.columns)}")

    df["mode"] = df["mode"].astype(str).str.strip().str.lower()

    assert len(df) == len(data), (
        f"标签行数({len(df)}) 与数据行数({len(data)}) 不匹配！"
    )

    logger.info(f"标签加载完成: {len(df)} 行 | 类别: {sorted(df['mode'].unique())}")

    _CACHE[npy_path] = (data, df)
    return data, df


# ──────────────────────────────────────────────────────────────────
#  IQ 复数信号 → STFT 幅度频谱图
# ──────────────────────────────────────────────────────────────────
def iq_to_spectrogram(iq: np.ndarray,
                      n_fft: int = 256,
                      hop_length: int = 64,
                      normalize: bool = True) -> np.ndarray:
    """复数 IQ 信号 → STFT 双边幅度频谱图 (dB)"""
    _, _, Zxx = stft(
        iq,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=False
    )
    spec = np.abs(Zxx).astype(np.float32)
    spec = np.fft.fftshift(spec, axes=0)   # 0Hz 移到中心行
    spec = 20.0 * np.log10(spec + 1e-8)   # 转 dB

    if normalize:
        lo, hi = spec.min(), spec.max()
        spec = (spec - lo) / (hi - lo + 1e-8)

    return spec


# ──────────────────────────────────────────────────────────────────
#  主数据集类（文件级无泄露切分）
# ──────────────────────────────────────────────────────────────────
class PanoradioDataset(Dataset):
    """
    RTL-SDR 开放集信号数据集 (按原始 512000 文件整体划分，防止时域泄漏)
    """
    def __init__(self, cfg: dict, mode: str = "train", transform=None):
        assert mode in ("train", "val", "test"), f"mode 必须是 train/val/test，得到: {mode}"

        self.cfg = cfg
        self.mode = mode
        self.transform = transform

        data_cfg = cfg["data"]
        feat_cfg = cfg["feature"]

        self.n_fft      = feat_cfg.get("n_fft", 256)
        self.hop_length = feat_cfg.get("hop_length", 64)
        self.normalize  = feat_cfg.get("normalize", True)

        self.target_class      = data_cfg["target_class"]
        self.known_negatives   = data_cfg.get("known_negatives", [])
        self.unknown_negatives = data_cfg.get("unknown_negatives", [])

        # 路径适配（根据实际文件名）
        raw_dir = data_cfg["raw_dir"]
        npy_name = data_cfg.get("npy_filename", "dataset_rtl.npy")
        tags_name = data_cfg.get("tags_filename", "datalabel_rtl.csv")
        
        npy_path  = os.path.join(raw_dir, npy_name)
        tags_path = os.path.join(raw_dir, tags_name)
        
        # 兼容旧名称兜底
        if not os.path.exists(npy_path):
            npy_path = os.path.join(raw_dir, "dataset_panoradio_hf.npy")
            tags_path = os.path.join(raw_dir, "dataset_panoradio_hf_tags.csv")

        self._data, self._df = _load_dataset(npy_path, tags_path)

        # 构建索引
        self._index: list[tuple[int, int, str]] = []
        self._build_index_file_level(data_cfg, mode)

        n_pos = sum(1 for _, l, _ in self._index if l == LABEL_POSITIVE)
        n_neg = len(self._index) - n_pos
        logger.info(
            f"[{mode.upper():5s}] 样本总数: {len(self._index):6d} "
            f"| 正类: {n_pos}  负类: {n_neg}"
        )

    def _build_index_file_level(self, data_cfg: dict, mode: str):
        """★ 关键：按原始 .npy 文件级别进行切分，彻底杜绝数据泄漏"""
        rng = np.random.default_rng(data_cfg.get("random_seed", 42))
        train_r = float(data_cfg.get("train_ratio", 0.7))
        val_r   = float(data_cfg.get("val_ratio", 0.15))
        chunks_per_file = data_cfg.get("chunks_per_file", 250)  # 512000 // 2048 = 250

        if mode in ("train", "val"):
            aliases = [self.target_class] + self.known_negatives
        else:
            aliases = [self.target_class] + self.known_negatives + self.unknown_negatives

        for alias in aliases:
            mask = self._df["mode"].apply(lambda m: _mode_matches(m, alias))
            rows = self._df.index[mask].to_numpy()

            if len(rows) == 0:
                logger.warning(f"  ⚠ 类别 '{alias}' 未匹配到任何样本！")
                continue

            # 计算包含多少个完整的原始 .npy 文件
            n_files = len(rows) // chunks_per_file
            if n_files == 0:
                # 兼容小于250切片的小样本模式
                file_ids = np.array([0])
                chunks_this = len(rows)
            else:
                file_ids = np.arange(n_files)
                chunks_this = chunks_per_file

            # 随机打乱文件顺序
            rng.shuffle(file_ids)

            n_train_files = max(1, int(len(file_ids) * train_r))
            n_val_files   = max(1, int(len(file_ids) * val_r))

            if mode == "train":
                sel_files = file_ids[:n_train_files]
            elif mode == "val":
                sel_files = file_ids[n_train_files: n_train_files + n_val_files]
            else:  # test
                sel_files = file_ids[n_train_files + n_val_files:]
                if len(sel_files) == 0:
                    sel_files = file_ids  # 若文件过少，降级全部作为测试

            label = LABEL_POSITIVE if _mode_matches(alias, self.target_class) else LABEL_NEGATIVE

            for fid in sel_files:
                start_idx = fid * chunks_this
                for offset in range(chunks_this):
                    if start_idx + offset < len(rows):
                        global_row = rows[start_idx + offset]
                        self._index.append((int(global_row), label, alias))

    # ── PyTorch Dataset 必须实现的两个核心接口 ──────────────────────────
    def __len__(self) -> int:
        """返回当前 split 的总样本数"""
        return len(self._index)

    def __getitem__(self, idx: int):
        """根据索引读取单个切片数据并转换为频谱图"""
        row_idx, label, cls_name = self._index[idx]

        iq = self._data[row_idx]  # (2048,) complex64

        spec = iq_to_spectrogram(
            iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalize=self.normalize,
        )

        # 增加通道维度: (1, n_fft, time_frames)
        x = torch.from_numpy(spec).unsqueeze(0)

        if self.transform:
            x = self.transform(x)

        return x, torch.tensor(label, dtype=torch.long), cls_name


# ──────────────────────────────────────────────────────────────────
#  DataLoader 构建工厂
# ──────────────────────────────────────────────────────────────────
def build_dataloaders(cfg: dict):
    bs = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)

    train_ds = PanoradioDataset(cfg, mode="train")
    val_ds   = PanoradioDataset(cfg, mode="val")
    test_ds  = PanoradioDataset(cfg, mode="test")

    def _make(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return _make(train_ds, True), _make(val_ds, False), _make(test_ds, False)