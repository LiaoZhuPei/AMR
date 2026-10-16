"""
build_dataset.py
-----------------
将 twin_droneRF 原始 mat 数据集切片打包为 npy 格式。
数据与标签存储在同一个 npy 文件中(结构化数组):
    dtype = [('data', float32, (2, M)), ('label', '<U16')]

目录结构假设:
twin_droneRF/
    train/  A1_IN_S0_slice_1.mat ...
    test/   0.mat, 1.mat ...
    test_labels.txt    (必需，用于反查 test/id.mat 对应的真实标签)
"""

import os
import re
import random
import argparse
from collections import defaultdict

import numpy as np
from scipy.io import loadmat, whosmat

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================
# 基础工具函数
# ============================================================

LABEL_RE = re.compile(r'^([A-Za-z]+)')


def parse_label(filename: str) -> str:
    """从原始文件名中解析类别标签(合并个体编号)。
    支持:
        'A1_IN_S0_slice_1.mat'    -> 'A'
        'background_slice_1.mat' -> 'BACKGROUND'
        'D3_IN_S0_slice_2.mat'   -> 'D'
    """
    stem = os.path.basename(filename)
    first_token = stem.split('_')[0]
    m = LABEL_RE.match(first_token)
    if not m:
        raise ValueError(f"无法从文件名解析标签: {filename}")
    return m.group(1).upper()


def _find_key(keys, target):
    for k in keys:
        if k.lower() == target.lower():
            return k
    return None


def get_sample_length(filepath: str, var_name='I') -> int:
    try:
        info = whosmat(filepath)
        names = [n for n, _, _ in info]
        key = _find_key(names, var_name)
        if key is None:
            raise KeyError(f"{var_name} 未在 {filepath} 中找到, 可用变量: {names}")
        for n, shape, _ in info:
            if n == key:
                return max(shape)
    except NotImplementedError:
        if not _HAS_H5PY:
            raise RuntimeError("检测到 v7.3 mat 文件，需要安装 h5py: pip install h5py")
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            key = _find_key(keys, var_name)
            return max(f[key].shape)


def load_iq(filepath: str):
    try:
        mat = loadmat(filepath)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        i_key = _find_key(keys, 'I')
        q_key = _find_key(keys, 'Q')
        I = np.asarray(mat[i_key]).astype(np.float32).ravel()
        Q = np.asarray(mat[q_key]).astype(np.float32).ravel()
    except NotImplementedError:
        if not _HAS_H5PY:
            raise RuntimeError("检测到 v7.3 mat 文件，需要安装 h5py: pip install h5py")
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            i_key = _find_key(keys, 'I')
            q_key = _find_key(keys, 'Q')
            I = np.asarray(f[i_key]).astype(np.float32).ravel()
            Q = np.asarray(f[q_key]).astype(np.float32).ravel()
    return I, Q


def compute_slice_positions(L, M, step, max_segments, strategy='uniform', rng=None):
    if L < M:
        return []
    total_possible = 1 + (L - M) // step

    if max_segments is None or max_segments <= 0 or total_possible <= max_segments:
        idx = np.arange(total_possible)
    else:
        keep = max_segments
        rng = rng or random
        if strategy == 'uniform':
            idx = np.linspace(0, total_possible - 1, keep)
            idx = np.unique(idx.astype(int))
            if len(idx) < keep:
                remain = list(set(range(total_possible)) - set(idx.tolist()))
                extra = rng.sample(remain, keep - len(idx))
                idx = np.array(sorted(list(idx) + extra))
        elif strategy == 'first':
            idx = np.arange(keep)
        elif strategy == 'random':
            idx = np.array(sorted(rng.sample(range(total_possible), keep)))
        else:
            raise ValueError(f"未知采样策略: {strategy}")
    return (idx * step).tolist()


def slice_iq(I, Q, M, positions):
    n = len(positions)
    out = np.empty((n, 2, M), dtype=np.float32)
    for j, s in enumerate(positions):
        out[j, 0, :] = I[s:s + M]
        out[j, 1, :] = Q[s:s + M]
    return out


# ============================================================
# 文件列表构建
# ============================================================

def list_train_files(train_dir, exclude_labels):
    files = []
    for fn in sorted(os.listdir(train_dir)):
        if not fn.lower().endswith('.mat'):
            continue
        label = parse_label(fn)
        if label in exclude_labels:
            continue
        files.append((os.path.join(train_dir, fn), label))
    return files


def list_test_files(test_dir, test_labels_txt, exclude_labels):
    id2label = {}
    with open(test_labels_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            orig_name, idx = parts[0], parts[1]
            id2label[idx] = parse_label(orig_name)

    def sort_key(fn):
        stem = os.path.splitext(fn)[0]
        return int(stem) if stem.isdigit() else stem

    files = []
    for fn in sorted(os.listdir(test_dir), key=sort_key):
        if not fn.lower().endswith('.mat'):
            continue
        idx = os.path.splitext(fn)[0]
        label = id2label.get(idx)
        if label is None:
            print(f"[警告] 未在 test_labels.txt 中找到 {fn} 对应标签, 已跳过")
            continue
        if label in exclude_labels:
            continue
        files.append((os.path.join(test_dir, fn), label))
    return files


def split_files(file_label_list, val_ratio, seed=42):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for item in file_label_list:
        by_label[item[1]].append(item)

    train_list, val_list = [], []
    for label, items in by_label.items():
        items = items[:]
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        val_list.extend(items[:n_val])
        train_list.extend(items[n_val:])

    rng.shuffle(train_list)
    rng.shuffle(val_list)
    return train_list, val_list


# ============================================================
# 核心构建函数
# ============================================================

def estimate_counts(file_label_list, M, overlap, max_segments):
    step = max(1, int(round(M * (1 - overlap))))
    total = 0
    per_file_counts = []
    for fp, _ in file_label_list:
        L = get_sample_length(fp)
        if L < M:
            per_file_counts.append(0)
            continue
        total_possible = 1 + (L - M) // step
        keep = total_possible if (max_segments is None or max_segments <= 0) \
            else min(total_possible, max_segments)
        per_file_counts.append(keep)
        total += keep
    return total, per_file_counts, step


def build_npy(file_label_list, M, overlap, max_segments,
              out_path,
              strategy='uniform', seed=42,
              byte_limit_gb=20.0, force=False,
              dtype=np.float32, label_dtype='<U16'):
    """
    核心函数: 扫描 -> 预估大小 -> 写入 单个结构化 npy 文件(同时含 data 和 label)

    输出的 npy 文件, 加载后是一个结构化数组 arr, 形状为 (N,)
        arr['data']  -> (N, 2, M)
        arr['label'] -> (N,)  字符串标签
    """
    if len(file_label_list) == 0:
        raise RuntimeError("文件列表为空，请检查目录/排除类别设置")

    rng = random.Random(seed)
    total_n, per_file_counts, step = estimate_counts(file_label_list, M, overlap, max_segments)

    struct_dtype = np.dtype([('data', dtype, (2, M)), ('label', label_dtype)])
    bytes_per_sample = struct_dtype.itemsize
    total_bytes = total_n * bytes_per_sample

    print(f"[信息] 目标文件: {out_path}")
    print(f"[信息] 预计样本数: {total_n}, 预计磁盘占用: {total_bytes / 1e9:.2f} GB "
          f"(M={M}, dtype={dtype})")

    if total_n == 0:
        raise RuntimeError("没有可用样本，请检查过滤条件 / M 是否大于原始序列长度")

    if total_bytes > byte_limit_gb * 1e9 and not force:
        raise MemoryError(
            f"预计生成数据超过设定阈值 {byte_limit_gb} GB。\n"
            f"可选方案: 减小 max_segments_per_file / 减小 M / 使用 float16 / "
            f"或设置 force=True(--force) 忽略此保护。"
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)

    # 结构化数组的磁盘映射写入: 数据 + 标签在同一个 npy 文件中
    arr_mm = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=struct_dtype, shape=(total_n,)
    )

    ptr = 0
    pbar = tqdm(list(zip(file_label_list, per_file_counts)),
                desc=f"生成 {os.path.basename(out_path)}")
    for (fp, label), keep in pbar:
        if keep == 0:
            continue
        I, Q = load_iq(fp)
        L = len(I)
        positions = compute_slice_positions(L, M, step, keep, strategy=strategy, rng=rng)
        seg = slice_iq(I, Q, M, positions).astype(dtype, copy=False)
        n = seg.shape[0]
        arr_mm['data'][ptr:ptr + n] = seg
        arr_mm['label'][ptr:ptr + n] = label
        ptr += n

    arr_mm.flush()
    del arr_mm

    print(f"[完成] {out_path}  样本数=({total_n},)  "
          f"data形状=(2,{M})  label为字符串")


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="twin_droneRF 数据集切片生成工具")
    parser.add_argument('--root', default='twin_droneRFa/twin_droneRF', help='数据集根目录')
    parser.add_argument('--M', type=int, default=2048, choices=[1024, 2048, 4096],
                         help='切片长度')
    parser.add_argument('--overlap', type=float, default=0.125, help='重叠率')
    parser.add_argument('--max_segments_per_file', type=int, default=400,
                         help='单个mat文件最多裁剪的片段数, <=0表示不限制')
    parser.add_argument('--exclude_train', nargs='*', default=['C', 'F'],
                         help='训练集(及验证集)需要排除的类别, 例如 --exclude_train C F')
    parser.add_argument('--exclude_test', nargs='*', default=[],
                         help='测试集需要排除的类别, 默认不排除任何类别(保留全部类别)')
    parser.add_argument('--val_ratio', type=float, default=0,
                         help='从train中划分验证集的比例(按文件级别划分), 0表示不划分')
    parser.add_argument('--strategy', default='uniform',
                         choices=['uniform', 'first', 'random'],
                         help='当片段数超过max_segments_per_file时的采样策略')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', default='./dataset_npy')
    parser.add_argument('--byte_limit_gb', type=float, default=20.0,
                         help='单个npy文件大小保护阈值(GB)，超过会报错提示')
    parser.add_argument('--force', action='store_true', help='忽略大小保护继续生成')
    parser.add_argument('--fp16', action='store_true', help='使用float16存储以减小体积')
    args = parser.parse_args()

    dtype = np.float16 if args.fp16 else np.float32
    exclude_train = set(l.upper() for l in args.exclude_train)
    exclude_test = set(l.upper() for l in args.exclude_test)
    os.makedirs(args.out_dir, exist_ok=True)

    train_dir = os.path.join(args.root, 'train')
    test_dir = os.path.join(args.root, 'test')
    test_labels_txt = os.path.join(args.root, 'test_labels.txt')

    # -------------------- 训练/验证 (排除 exclude_train, 默认 C, F) --------------------
    print("========== 扫描训练集 ==========")
    train_files = list_train_files(train_dir, exclude_train)
    print(f"训练集(排除 {exclude_train} 后) 文件数: {len(train_files)}")

    if args.val_ratio > 0:
        tr_list, val_list = split_files(train_files, args.val_ratio, args.seed)
    else:
        tr_list, val_list = train_files, []

    build_npy(
        tr_list, args.M, args.overlap, args.max_segments_per_file,
        os.path.join(args.out_dir, f'train_M{args.M}.npy'),
        strategy=args.strategy, seed=args.seed,
        byte_limit_gb=args.byte_limit_gb, force=args.force, dtype=dtype
    )

    if val_list:
        build_npy(
            val_list, args.M, args.overlap, args.max_segments_per_file,
            os.path.join(args.out_dir, f'val_M{args.M}.npy'),
            strategy=args.strategy, seed=args.seed,
            byte_limit_gb=args.byte_limit_gb, force=args.force, dtype=dtype
        )

    # -------------------- 测试集 (排除 exclude_test, 默认不排除, 保留全部类别) --------------------
    print("========== 扫描测试集 ==========")
    test_files = list_test_files(test_dir, test_labels_txt, exclude_test)
    print(f"测试集(排除 {exclude_test if exclude_test else '无'} 后) 文件数: {len(test_files)}")

    build_npy(
        test_files, args.M, args.overlap, args.max_segments_per_file,
        os.path.join(args.out_dir, f'test_M{args.M}.npy'),
        strategy=args.strategy, seed=args.seed,
        byte_limit_gb=args.byte_limit_gb, force=args.force, dtype=dtype
    )


if __name__ == '__main__':
    main()