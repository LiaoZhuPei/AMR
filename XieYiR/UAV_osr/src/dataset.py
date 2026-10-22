"""
dataset.py
数据加载模块（多球体 Deep SVDD 版本）

数据格式:
  每个 .npy 文件保存为字典: {"data": (N, 1024) 原始电压序列, "labels": (N,) 类别名称字符串}

  - train_known.npy   : 目标类(4类) + 已知负类(3类)，内部按比例切分 train/val
  - test_known.npy    : 与 train_known 相同的类别体系，用于闭集评估 + 已知负类拒识评估
  - test_unknown.npy  : 全新未知类别(5类)，用于开放集拒识能力评估

类别体系:
  - target_classes:         需要建球的已知目标类别 (K个)，label = 0..K-1
                             例: BEEBEERUN, DJI_M600, DJI_INSPIRE, DJI_PHANTOM
  - known_negative_classes: 训练/测试中出现的已知非目标类别，label = -1
                             例: APPLE_IPAD3, FITBIT_CHARGE3, CISCO_LINKSYS_E3200
  - unknown_classes:        仅出现在 test_unknown.npy 中的全新未知类别，label = -1
                             例: DJI_MAVICPRO, APPLE_IPHONE6S, APPLE_IPHONE7,
                                 MOTOROLA, TPLINK_TL_WR940N
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
#  标签定义
# ──────────────────────────────────────────────────────────────────
LABEL_UNKNOWN = -1   # 已知负类 / 未知类 统一标记


# ──────────────────────────────────────────────────────────────────
#  .npy 字典文件加载（全局缓存）
# ──────────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _load_npy_dict(path: str) -> tuple:
    """
    加载字典型 .npy 文件: {"data": (N,1024), "labels": (N,)}
    支持常见的 key 别名，兼容不同保存习惯。
    """
    if path in _CACHE:
        return _CACHE[path]

    if not os.path.exists(path):
        raise FileNotFoundError(f"\n[数据缺失] 找不到: {path}")

    logger.info(f"正在加载数据: {path} ...")
    raw = np.load(path, allow_pickle=True)

    # np.save 保存 dict 时会被包装成 0-d object array
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.item()

    if not isinstance(raw, dict):
        raise ValueError(f"{path} 不是预期的字典格式，实际类型: {type(raw)}")

    data_keys = ["data", "X", "signals", "samples"]
    label_keys = ["labels", "label", "y", "tags"]

    data = None
    for k in data_keys:
        if k in raw:
            data = np.asarray(raw[k])
            break
    if data is None:
        raise KeyError(f"{path} 中未找到数据字段，尝试的键: {data_keys}，"
                        f"实际键: {list(raw.keys())}")

    labels = None
    for k in label_keys:
        if k in raw:
            labels = np.asarray(raw[k])
            break
    if labels is None:
        raise KeyError(f"{path} 中未找到标签字段，尝试的键: {label_keys}，"
                        f"实际键: {list(raw.keys())}")

    labels = np.array([str(l).strip() for l in labels])

    assert len(data) == len(labels), (
        f"{path}: 数据行数({len(data)}) 与标签行数({len(labels)}) 不匹配！"
    )

    logger.info(f"数据加载完成: shape={data.shape}, dtype={data.dtype} | "
                f"类别: {sorted(set(labels.tolist()))}")

    _CACHE[path] = (data, labels)
    return data, labels


# ──────────────────────────────────────────────────────────────────
#  信号归一化（原始电压值幅值范围差异较大，需归一化后再送入网络）
# ──────────────────────────────────────────────────────────────────
def normalize_signal(sig: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    对单条原始电压序列做归一化。

    Args:
        sig:    (signal_length,) 原始电压值，如 [7312, -8700, -20344, ...]
        method: "zscore" | "minmax" | "none"
    """
    sig = sig.astype(np.float32)
    if method == "zscore":
        mean = sig.mean()
        std = sig.std()
        sig = (sig - mean) / (std + 1e-8)
    elif method == "minmax":
        lo, hi = sig.min(), sig.max()
        sig = (sig - lo) / (hi - lo + 1e-8) * 2.0 - 1.0  # 归一化到 [-1, 1]
    elif method in ("none", None):
        pass
    else:
        raise ValueError(f"Unknown normalize method: {method}")
    return sig


# ──────────────────────────────────────────────────────────────────
#  主数据集类
# ──────────────────────────────────────────────────────────────────
class DroneRFDataset(Dataset):
    """
    无人机/干扰源射频信号 开放集识别数据集（多球体 Deep SVDD 用）。

    文件划分:
      - train_known.npy  → 内部按类别随机切分为 train / val
      - test_known.npy   → 全部作为 test 的一部分（闭集评估 + 已知负类拒识评估）
      - test_unknown.npy → 全部作为 test 的一部分（开放集拒识评估）

    注意:
      若原始数据是由更长录音切片而成，同一录音的不同切片可能同时落入
      train 和 val，存在时域泄漏风险。若 .npy 字典中提供了 "groups" 字段
      （标识每条样本所属的原始录音/文件），可在 _build_train_val_index 中
      改为按组切分；当前默认按样本级随机切分。
    """

    def __init__(self, cfg: dict, mode: str = "train", transform=None):
        assert mode in ("train", "val", "test"), f"mode 必须是 train/val/test，得到: {mode}"

        self.cfg = cfg
        self.mode = mode
        self.transform = transform

        data_cfg = cfg["data"]
        feat_cfg = cfg.get("feature", {})
        self.normalize_method = feat_cfg.get("normalize_signal", "zscore")

        # 类别体系（顺序决定 target_classes 的数字标签 0..K-1）
        self.target_classes = data_cfg["target_classes"]
        self.known_negative_classes = data_cfg.get("known_negative_classes", [])
        self.unknown_classes = data_cfg.get("unknown_classes", [])
        self.class_to_idx = {name: i for i, name in enumerate(self.target_classes)}

        raw_dir = data_cfg["raw_dir"]
        self._sources: dict = {}                 # source_key -> np.ndarray (N, 1024)
        self._index: list = []                    # (source_key, row_idx, label, cls_name)

        if mode in ("train", "val"):
            npy_path = os.path.join(
                raw_dir, data_cfg.get("train_known_filename", "train_known.npy")
            )
            data, labels_raw = _load_npy_dict(npy_path)
            self._sources["train_known"] = data
            self._build_train_val_index(labels_raw, data_cfg, mode, source_key="train_known")

        else:  # test
            known_path = os.path.join(
                raw_dir, data_cfg.get("test_known_filename", "test_known.npy")
            )
            unknown_path = os.path.join(
                raw_dir, data_cfg.get("test_unknown_filename", "test_unknown.npy")
            )

            data_known, labels_known_raw = _load_npy_dict(known_path)
            self._sources["test_known"] = data_known
            self._build_test_known_index(labels_known_raw, source_key="test_known")

            data_unknown, labels_unknown_raw = _load_npy_dict(unknown_path)
            self._sources["test_unknown"] = data_unknown
            self._build_test_unknown_index(labels_unknown_raw, source_key="test_unknown")

        self._log_summary()

    # ─────────────────────────────────────────────
    #  索引构建
    # ─────────────────────────────────────────────

    def _resolve_label(self, cls_name: str) -> int:
        """已知目标类 → 0..K-1；已知负类 → -1"""
        if cls_name in self.class_to_idx:
            return self.class_to_idx[cls_name]
        return LABEL_UNKNOWN

    def _build_train_val_index(self, labels_raw: np.ndarray, data_cfg: dict,
                               mode: str, source_key: str):
        """
        对 train_known.npy 中的每个类别（目标类 + 已知负类）分别做随机切分。
        """
        rng = np.random.default_rng(data_cfg.get("random_seed", 42))
        train_ratio = float(data_cfg.get("train_ratio", 0.8))

        all_classes = self.target_classes + self.known_negative_classes
        for cls_name in all_classes:
            rows = np.where(labels_raw == cls_name)[0]
            if len(rows) == 0:
                logger.warning(f"  ⚠ 类别 '{cls_name}' 在 train_known.npy 中未匹配到任何样本！")
                continue

            rows = rows.copy()
            rng.shuffle(rows)
            n_train = max(1, int(len(rows) * train_ratio))

            if mode == "train":
                sel_rows = rows[:n_train]
            else:  # val
                sel_rows = rows[n_train:]
                if len(sel_rows) == 0:
                    # 极小类别兜底，至少留一部分做验证
                    sel_rows = rows[-max(1, len(rows) // 10):]

            label = self._resolve_label(cls_name)
            for r in sel_rows:
                self._index.append((source_key, int(r), label, cls_name))

    def _build_test_known_index(self, labels_raw: np.ndarray, source_key: str):
        """test_known.npy 全部样本进入测试集"""
        all_classes = self.target_classes + self.known_negative_classes
        for cls_name in all_classes:
            rows = np.where(labels_raw == cls_name)[0]
            if len(rows) == 0:
                logger.warning(f"  ⚠ 类别 '{cls_name}' 在 test_known.npy 中未匹配到任何样本！")
                continue
            label = self._resolve_label(cls_name)
            for r in rows:
                self._index.append((source_key, int(r), label, cls_name))

    def _build_test_unknown_index(self, labels_raw: np.ndarray, source_key: str):
        """test_unknown.npy 全部样本标记为 -1（真·未知类）"""
        for cls_name in self.unknown_classes:
            rows = np.where(labels_raw == cls_name)[0]
            if len(rows) == 0:
                logger.warning(f"  ⚠ 类别 '{cls_name}' 在 test_unknown.npy 中未匹配到任何样本！")
                continue
            for r in rows:
                self._index.append((source_key, int(r), LABEL_UNKNOWN, cls_name))

    def _log_summary(self):
        n_total = len(self._index)
        per_class_count = {}
        for _, _, _, cls_name in self._index:
            per_class_count[cls_name] = per_class_count.get(cls_name, 0) + 1

        logger.info(f"[{self.mode.upper():5s}] 样本总数: {n_total}")
        for cls_name, cnt in sorted(per_class_count.items()):
            if cls_name in self.class_to_idx:
                role = f"目标类(label={self.class_to_idx[cls_name]})"
            elif cls_name in self.known_negative_classes:
                role = "已知负类(label=-1)"
            else:
                role = "未知类(label=-1)"
            logger.info(f"    [{role}] {cls_name}: {cnt}")

    # ─────────────────────────────────────────────
    #  PyTorch Dataset 接口
    # ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        source_key, row_idx, label, cls_name = self._index[idx]
        sig = self._sources[source_key][row_idx]  # (1024,) 原始电压值

        sig = normalize_signal(sig, method=self.normalize_method)

        # 增加通道维度: (1, signal_length)
        x = torch.from_numpy(sig).unsqueeze(0).float()

        if self.transform:
            x = self.transform(x)

        return x, torch.tensor(label, dtype=torch.long), cls_name


# ──────────────────────────────────────────────────────────────────
#  DataLoader 构建工厂
# ──────────────────────────────────────────────────────────────────
def build_dataloaders(cfg: dict):
    """
    构建 train / val / test 三个 DataLoader。

    test DataLoader 内部同时包含 test_known.npy + test_unknown.npy 的样本，
    label 已统一为: 目标类 0..K-1 / 已知负类及未知类 -1，
    cls_name 保留原始类别字符串，供 Evaluator 做分组统计与绘图。
    """
    bs = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)

    train_ds = DroneRFDataset(cfg, mode="train")
    val_ds   = DroneRFDataset(cfg, mode="val")
    test_ds  = DroneRFDataset(cfg, mode="test")

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