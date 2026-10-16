"""
dataset.py
数据加载模块 - 适配 IQ 数据格式

数据格式:
  .npy 文件的 dtype: [('data', '<f4', (2, 2048)), ('label', '<U16')]
    - data:  (2, 2048)  第一行=I, 第二行=Q
    - label: 字符串,如 'A','B','C','D','E','F','G'

  - train_M2048.npy: 训练数据（不含 C, F）
  - test_M2048.npy : 测试数据（含全部 A-G）

类别体系（由外部配置决定，本模块只按类别名索引）:
  - target_classes:  目标类别列表, label = 0..K-1
  - unknown_classes: 未知/负类别列表, label = -1
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

LABEL_UNKNOWN = -1


def _load_npy(path: str):
    """加载结构化 .npy 文件, 返回 (data (N,2,L), labels (N,) str)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件: {path}")
    arr = np.load(path, allow_pickle=True)
    data = np.asarray(arr["data"], dtype=np.float32)
    labels = np.array([str(l).strip() for l in arr["label"]])
    return data, labels


def normalize_iq(sig: np.ndarray, method: str = "zscore") -> np.ndarray:
    """对 IQ 信号做归一化, sig shape=(2, L)"""
    sig = sig.astype(np.float32)
    if method == "zscore":
        mean = sig.mean(axis=1, keepdims=True)
        std = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - mean) / std
    elif method == "minmax":
        lo = sig.min(axis=1, keepdims=True)
        hi = sig.max(axis=1, keepdims=True)
        sig = (sig - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    return sig


class IQDataset(Dataset):
    """
    IQ 信号开放集识别数据集。

    Args:
        npy_path:         .npy 文件路径
        target_classes:   目标类别名列表, 标签 0..K-1
        unknown_classes:  未知/负类别名列表, 标签 -1
                          若为 None, 则自动把所有非 target 的类别归为 -1
        indices:          仅使用指定行下标 (用于 train/val 切分)
        normalize:        归一化方式 "zscore" | "minmax" | "none"
    """

    def __init__(self, npy_path, target_classes,
                 unknown_classes=None, indices=None,
                 normalize="zscore"):
        self.data, self.labels_raw = _load_npy(npy_path)
        self.target_classes = list(target_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}
        self.normalize = normalize

        # 未知类：显式给定则只保留其中的类别; 否则所有非目标类都当未知
        all_cls = set(np.unique(self.labels_raw).tolist())
        if unknown_classes is None:
            self.unknown_classes = sorted(all_cls - set(self.target_classes))
        else:
            self.unknown_classes = list(unknown_classes)

        valid_cls = set(self.target_classes) | set(self.unknown_classes)

        # 构建索引: (row_idx, label_int, cls_name)
        rows = np.arange(len(self.labels_raw)) if indices is None else np.asarray(indices)
        self._index = []
        for r in rows:
            cls = self.labels_raw[r]
            if cls not in valid_cls:
                continue
            lab = self.class_to_idx.get(cls, LABEL_UNKNOWN)
            self._index.append((int(r), lab, cls))

    def get_class_distribution(self):
        """返回各类别样本数, 用于日志输出"""
        dist = {}
        for _, _, cls in self._index:
            dist[cls] = dist.get(cls, 0) + 1
        return dist

    def get_indices_by_class(self):
        """按类别返回行下标 dict, 用于外部 train/val 切分"""
        d = {}
        for r, _, cls in self._index:
            d.setdefault(cls, []).append(r)
        return d

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        row, label, cls = self._index[idx]
        sig = normalize_iq(self.data[row], self.normalize)  # (2, L)
        x = torch.from_numpy(sig).float()                    # (2, L)
        return x, torch.tensor(label, dtype=torch.long), cls


def split_train_val(npy_path, target_classes, unknown_classes,
                    train_ratio=0.8, seed=42, normalize="zscore"):
    """
    从同一个训练 .npy 中按类别分层随机切分 train / val 数据集。
    返回: (train_ds, val_ds)
    """
    # 先构建全量, 用其索引信息做切分
    full = IQDataset(npy_path, target_classes, unknown_classes,
                     indices=None, normalize=normalize)
    per_cls = full.get_indices_by_class()

    rng = np.random.default_rng(seed)
    train_rows, val_rows = [], []
    for cls, rows in per_cls.items():
        rows = np.array(rows)
        rng.shuffle(rows)
        n_tr = max(1, int(len(rows) * train_ratio))
        train_rows.extend(rows[:n_tr].tolist())
        val_rows.extend(rows[n_tr:].tolist() if len(rows) > n_tr else rows[-1:].tolist())

    train_ds = IQDataset(npy_path, target_classes, unknown_classes,
                         indices=train_rows, normalize=normalize)
    val_ds = IQDataset(npy_path, target_classes, unknown_classes,
                       indices=val_rows, normalize=normalize)
    return train_ds, val_ds


def build_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))