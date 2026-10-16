"""
dataset.py
数据加载模块 - 适配大文件 IQ 数据格式（带内存映射与全局缓存防爆内存版）

数据格式:
  .npy dtype: [('data', '<f4', (2, 2048)), ('label', '<U16')]
    - data:  (2, 2048)  I/Q 双通道
    - label: 字符串 A-G
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 全局缓存字典：确保 9.17 GiB 的大文件在内存中只被映射一次
_CACHE = {}


def _load_npy(path: str):
    """
    加载结构化 .npy 文件（内存映射版）
    使用 mmap_mode='r' 避免 9GB 数据直接挤爆 RAM
    """
    if path in _CACHE:
        return _CACHE[path]

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件: {path}")

    print(f"[数据加载] 正在建立磁盘内存映射: {path} ...")

    # 核心：使用 mmap_mode='r'，NumPy 只读取文件头，不加载实际数据到内存
    arr = np.load(path, mmap_mode='r', allow_pickle=True)

    # data 依然保持内存映射状态 (Memory-mapped ndarray)
    data = arr["data"]

    # 标签数据通常很小 (60万个字符约占 30MB)，可以直接读入物理内存方便快速检索和比对
    print(f"[数据加载] 正在解析标签索引...")
    labels = np.array([str(l).strip() for l in arr["label"]])

    _CACHE[path] = (data, labels)
    print(f"[数据加载] 磁盘映射建立完成！ shape={data.shape}")
    return data, labels


def normalize_iq(sig: np.ndarray, method: str = "zscore") -> np.ndarray:
    """归一化 IQ 信号, sig shape=(2, L)"""
    sig = sig.astype(np.float32)
    if method == "zscore":
        m = sig.mean(axis=1, keepdims=True)
        s = sig.std(axis=1, keepdims=True) + 1e-8
        sig = (sig - m) / s
    elif method == "minmax":
        lo = sig.min(axis=1, keepdims=True)
        hi = sig.max(axis=1, keepdims=True)
        sig = (sig - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    return sig


# ─────────────────────────────────────────────
#  多类数据集 (evaluate 用)
# ─────────────────────────────────────────────

class IQDataset(Dataset):
    """
    多类模式: target_classes 中的类别 → 0..K-1, 其余 → -1
    """

    def __init__(self, npy_path, target_classes, unknown_classes=None,
                 indices=None, normalize="zscore"):
        self.data, self.labels_raw = _load_npy(npy_path)
        self.target_classes = list(target_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}
        self.normalize = normalize

        all_cls = set(np.unique(self.labels_raw))
        self.unknown_classes = list(unknown_classes) if unknown_classes else \
            sorted(all_cls - set(self.target_classes))
        valid = set(self.target_classes) | set(self.unknown_classes)

        rows = np.arange(len(self.labels_raw)) if indices is None else np.asarray(indices)
        self._index = []
        for r in rows:
            c = self.labels_raw[r]
            if c not in valid:
                continue
            self._index.append((int(r), self.class_to_idx.get(c, -1), c))

    def get_class_distribution(self):
        d = {}
        for _, _, c in self._index:
            d[c] = d.get(c, 0) + 1
        return d

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        r, lab, cls = self._index[idx]
        # 此时 self.data[r] 会触发极为快速的磁盘随机 I/O，只读取该行 16KB 左右的数据
        sig = normalize_iq(np.array(self.data[r]), self.normalize)
        return torch.from_numpy(sig).float(), torch.tensor(lab, dtype=torch.long), cls


# ─────────────────────────────────────────────
#  单目标二分类数据集 (train 用)
# ─────────────────────────────────────────────

class BinarySphereDataset(Dataset):
    """
    单目标二分类模式:
      target_class (如 "A") → label=1 (目标)
      其余 allowed_negatives 中的类别 → label=0 (负类)
    """

    def __init__(self, npy_path, target_class, negative_classes,
                 indices=None, normalize="zscore"):
        self.data, self.labels_raw = _load_npy(npy_path)
        self.target_class = target_class
        self.negative_classes = list(negative_classes)
        self.normalize = normalize

        valid = {target_class} | set(self.negative_classes)
        rows = np.arange(len(self.labels_raw)) if indices is None else np.asarray(indices)
        self._index = []
        for r in rows:
            c = self.labels_raw[r]
            if c not in valid:
                continue
            binary_label = 1 if c == target_class else 0
            self._index.append((int(r), binary_label, c))

    def get_class_distribution(self):
        d = {}
        for _, _, c in self._index:
            d[c] = d.get(c, 0) + 1
        return d

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        r, lab, cls = self._index[idx]
        # 此时只从大磁盘映射文件中加载这单条 IQ 样本，物理内存占用极低
        sig = normalize_iq(np.array(self.data[r]), self.normalize)
        return torch.from_numpy(sig).float(), torch.tensor(lab, dtype=torch.long), cls


# ─────────────────────────────────────────────
#  切分工具
# ─────────────────────────────────────────────

def split_binary_train_val(npy_path, target_class, negative_classes,
                           train_ratio=0.8, seed=42, normalize="zscore"):
    """
    从同一 .npy 按类别分层切分 train/val, 返回两个 BinarySphereDataset。
    因为引入了全局缓存，创建 `full`、`tr`、`va` 时不会重复读取物理文件，只做指针引用。
    """
    full = BinarySphereDataset(npy_path, target_class, negative_classes,
                               normalize=normalize)
    per_cls = {}
    for r, _, c in full._index:
        per_cls.setdefault(c, []).append(r)

    rng = np.random.default_rng(seed)
    tr_rows, va_rows = [], []
    for cls, rows in per_cls.items():
        rows = np.array(rows)
        rng.shuffle(rows)
        n = max(1, int(len(rows) * train_ratio))
        tr_rows.extend(rows[:n].tolist())
        va_rows.extend(rows[n:].tolist() if len(rows) > n else rows[-1:].tolist())

    tr = BinarySphereDataset(npy_path, target_class, negative_classes,
                             indices=tr_rows, normalize=normalize)
    va = BinarySphereDataset(npy_path, target_class, negative_classes,
                             indices=va_rows, normalize=normalize)
    return tr, va


def build_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))