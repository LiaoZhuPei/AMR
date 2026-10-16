#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_compare.py — 推理脚本（支持 .mat 文件 + .npz 生成数据集，含多场景对比）

使用方法:
    # 模式1: 对 .mat 文件批量推理（原有功能）
    python infer_compare.py --mode mat --folder /path/to/mat_files

    # 模式2: 对 npz 数据集推理并输出逐场景、逐SNR准确率
    python infer_compare.py --mode npz --npz ./multirate_dataset/amr_multirate.npz

    # 模式3: 两者都做
    python infer_compare.py --mode both --folder /path/to/mat --npz ./multirate_dataset/amr_multirate.npz
"""

import os
import glob
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ===== 导入模型 =====
from test01_model import build_model
# from model_tcn import build_model
# from model_transformer import  build_model
# ─────────────────────────────────────────────────────────────────────
# 基本配置（修改这里）
# ─────────────────────────────────────────────────────────────────────
CLASSES = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

MODEL_WEIGHTS_PATH = 'weights/cnn_bilstm_attn_multirate_2.pt'
# MODEL_WEIGHTS_PATH = 'weights/transformer_multirate_2.pt'
NUM_SAMPLES_MAT    = 100            # mat 模式最多读几个文件
BATCH_SIZE_NPZ     = 256            # npz 模式推理时的 batch size
OUTPUT_DIR         = './infer_results_cnnbilstm'   # 图表和 CSV 保存目录


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────
def power_normalize(signal_data: np.ndarray) -> np.ndarray:
    """
    功率归一化，与 train.py 中预处理完全一致。
    输入: (..., 2, 1024) 或 (2, 1024)
    """
    if signal_data.ndim == 3:
        I = signal_data[..., 0, :]
        Q = signal_data[..., 1, :]
    else:
        I = signal_data[0]
        Q = signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


def load_model(weights_path, n_classes, device):
    model = build_model(classes=n_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"  模型加载成功  参数量: {total:,}")
    return model


def predict_batch(model, x_batch: np.ndarray, device) -> np.ndarray:
    """
    批量推理。
    输入 x_batch: (N, 2, 1024)
    输出: (N,) int  预测类别索引
    """
    # 功率归一化
    pwr = np.mean(x_batch[:, 0] ** 2 + x_batch[:, 1] ** 2, axis=-1, keepdims=True)
    pwr = pwr[:, :, np.newaxis]    # (N,1,1)
    x_norm = x_batch / (np.sqrt(pwr) + 1e-10)

    # (N, 2, 1024) → (N, 1024, 2) 与 Dataset.__getitem__ 一致
    t = torch.from_numpy(x_norm).float()          # (N, 2, 1024)
    t = t.permute(0, 2, 1).to(device)             # (N, 1024, 2)

    with torch.no_grad():
        logits = model(t)                          # (N, num_classes)
    return logits.argmax(dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────
# 模式1: mat 文件推理（原有功能，仅修改接口）
# ─────────────────────────────────────────────────────────────────────
def run_mat_mode(model, folder, device):
    if not HAS_H5PY:
        print("[Error] h5py 未安装，无法读取 .mat 文件。pip install h5py")
        return

    pattern   = os.path.join(folder, '*.mat')
    all_files = glob.glob(pattern)
    if not all_files:
        print(f"[Error] 未找到 .mat 文件: {pattern}")
        return

    files_to_test = all_files[:NUM_SAMPLES_MAT]
    print(f"\n找到 {len(all_files)} 个 .mat 文件，测试前 {len(files_to_test)} 个")

    prediction_counts = Counter()
    print("\n" + "=" * 50)

    for i, fp in enumerate(files_to_test):
        try:
            with h5py.File(fp, 'r') as f:
                sig = f['data_save'][:]          # 期望 (1, 2, 1024)

            if sig.shape != (1, 2, 1024):
                print(f"  [跳过] {os.path.basename(fp)} 形状 {sig.shape}")
                continue

            pred_idx = predict_batch(model, sig, device)[0]   # (1,2,1024) → scalar
            pred_cls = CLASSES[pred_idx]
            prediction_counts[pred_cls] += 1
            print(f"  [{i+1:3d}] {os.path.basename(fp):<40s} → {pred_cls}")

        except Exception as e:
            print(f"  [Error] {os.path.basename(fp)}: {e}")

    # 摘要
    total = sum(prediction_counts.values())
    print("\n" + "=" * 50)
    print(f"  成功处理: {total}/{len(files_to_test)}")
    print("  预测分布:")
    for cls in CLASSES:
        cnt = prediction_counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"    {cls:<12}: {cnt:>4}  ({pct:5.1f}%)")

    most = prediction_counts.most_common(1)
    if most:
        print(f"\n  最多预测类别: {most[0][0]} ({most[0][1]} 次)")


# ─────────────────────────────────────────────────────────────────────
# 模式2: npz 数据集推理 + 逐场景逐SNR准确率对比
# ─────────────────────────────────────────────────────────────────────
def run_npz_mode(model, npz_path, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n加载数据集: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    X_test    = data['X_test']          # (N, 2, 1024)
    Y_test    = data['Y_test']          # (N,) int
    snr_test  = data['snr_test']        # (N,) float
    classes   = list(data['classes'])

    # 场景信息（旧版 npz 可能没有，做兼容）
    if 'scene_test' in data:
        scene_test   = data['scene_test']        # (N,) int
        sr_test      = data['sr_test']            # (N,) float  符号率
        scene_labels = list(data['scene_labels']) # 场景名称
        has_scene    = True
    else:
        print("  [注意] 数据集无场景字段（旧版单场景 npz），将视为单场景运行")
        scene_test   = np.zeros(len(Y_test), dtype=np.int64)
        sr_test      = np.ones(len(Y_test),  dtype=np.float32)
        scene_labels = ['default']
        has_scene    = False

    snr_vals   = sorted(set(snr_test.tolist()))
    scene_idxs = sorted(set(scene_test.tolist()))
    N = len(Y_test)

    print(f"  测试集: {N:,} 样本  "
          f"SNR: {snr_vals[0]:.0f}~{snr_vals[-1]:.0f} dB  "
          f"场景: {len(scene_idxs)}")

    # ── 批量推理 ──
    print("\n  开始推理...")
    preds = np.empty(N, dtype=np.int64)
    for start in range(0, N, BATCH_SIZE_NPZ):
        end   = min(start + BATCH_SIZE_NPZ, N)
        batch = X_test[start:end]
        preds[start:end] = predict_batch(model, batch, device)
        if (start // BATCH_SIZE_NPZ) % 20 == 0:
            print(f"    {end}/{N} ({end/N*100:.1f}%)")

    overall_acc = (preds == Y_test).mean()
    print(f"\n  整体准确率: {overall_acc*100:.2f}%")

    # ── 逐场景 × 逐SNR × 逐类 准确率表 ──
    records = []
    for sc in scene_idxs:
        sc_mask  = scene_test == sc
        sc_label = scene_labels[sc]

        for snr in snr_vals:
            snr_mask = snr_test == snr
            mask = sc_mask & snr_mask
            if not mask.any():
                continue

            y_true = Y_test[mask]
            y_pred = preds[mask]
            acc    = (y_pred == y_true).mean()

            row = {
                'scene':    sc_label,
                'scene_idx': sc,
                'snr_db':   snr,
                'overall_acc': acc,
                'n_samples': mask.sum(),
            }
            # 逐类准确率
            for ci, cname in enumerate(classes):
                cm = y_true == ci
                row[f'acc_{cname}'] = (y_pred[cm] == ci).mean() if cm.any() else float('nan')

            records.append(row)

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, 'scene_snr_accuracy.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n  准确率表已保存 → {csv_path}")
    print(df[['scene', 'snr_db', 'overall_acc']].to_string(index=False))

    # ── 绘图 ──
    if HAS_MPL:
        _plot_results(df, classes, scene_labels, scene_idxs, output_dir)


def _plot_results(df, classes, scene_labels, scene_idxs, output_dir):
    """绘制两张图: 整体 acc vs SNR，各类 acc vs SNR（多场景对比）"""

    # 图1: 整体准确率 vs SNR，按场景分色
    fig, ax = plt.subplots(figsize=(10, 5))
    for sc in scene_idxs:
        sub = df[df['scene_idx'] == sc]
        ax.plot(sub['snr_db'], sub['overall_acc'] * 100,
                '.-', label=scene_labels[sc], ms=4)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('Overall Accuracy vs SNR — Scene Comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    p1 = os.path.join(output_dir, 'overall_acc_vs_snr.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  图1 → {p1}")

    # 图2: 逐类准确率 vs SNR（仅高SNR段，≥10dB），各场景子图
    focus_classes = ['2FSK', '4FSK', '16QAM', 'BPSK', 'QPSK', 'GMSK']
    n_sc   = len(scene_idxs)
    n_cols = min(n_sc, 3)
    n_rows = (n_sc + n_cols - 1) // n_cols

    fig2, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 4 * n_rows),
                              squeeze=False)
    for si, sc in enumerate(scene_idxs):
        ax = axes[si // n_cols, si % n_cols]
        sub = df[df['scene_idx'] == sc]
        for cname in focus_classes:
            col = f'acc_{cname}'
            if col in sub.columns:
                ax.plot(sub['snr_db'], sub[col] * 100,
                        '.-', label=cname, ms=3)
        ax.set_title(f'Scene: {scene_labels[sc]}', fontsize=9)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    # 隐藏多余子图
    for si in range(len(scene_idxs), n_rows * n_cols):
        axes[si // n_cols, si % n_cols].set_visible(False)

    fig2.suptitle('Per-Class Accuracy vs SNR by Scene', fontsize=11)
    fig2.tight_layout()
    p2 = os.path.join(output_dir, 'per_class_acc_by_scene.png')
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)
    print(f"  图2 → {p2}")

    # 图3: 重点类 2FSK/4FSK/16QAM 跨场景对比（所有场景画在一张图）
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    target_classes = ['2FSK', '4FSK', '16QAM']
    for ai, cname in enumerate(target_classes):
        ax = axes3[ai]
        for sc in scene_idxs:
            sub = df[df['scene_idx'] == sc]
            col = f'acc_{cname}'
            if col in sub.columns:
                ax.plot(sub['snr_db'], sub[col] * 100,
                        '.-', label=scene_labels[sc], ms=4)
        ax.set_title(f'{cname} — Scene Comparison')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    fig3.tight_layout()
    p3 = os.path.join(output_dir, 'key_class_scene_compare.png')
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)
    print(f"  图3 → {p3}")


# ─────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='AMR 推理：支持 mat 文件 和 npz 数据集两种模式'
    )
    parser.add_argument('--mode',   choices=['mat', 'npz', 'both'], default='npz',
                        help='推理模式 (default: npz)')
    parser.add_argument('--folder', default='E:/Project/AMR-Benchmark/output_mat_files_GMSK9/',
                        help='mat 文件目录')
    parser.add_argument('--npz',    default='./multirate_dataset/amr_multirate_test.npz',
                        help='npz 数据集路径')
    parser.add_argument('--weights', default=MODEL_WEIGHTS_PATH,
                        help='模型权重路径')
    parser.add_argument('--output', default=OUTPUT_DIR,
                        help='结果输出目录')
    return parser.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    # 加载模型
    if not os.path.exists(args.weights):
        print(f"[Error] 模型权重不存在: {args.weights}")
        exit(1)

    print(f"\n加载模型权重: {args.weights}")
    model = load_model(args.weights, len(CLASSES), device)

    # 执行推理
    if args.mode in ('mat', 'both'):
        print("\n" + "=" * 60)
        print("  模式: .mat 文件批量推理")
        print("=" * 60)
        run_mat_mode(model, args.folder, device)

    if args.mode in ('npz', 'both'):
        print("\n" + "=" * 60)
        print("  模式: npz 数据集推理 + 场景对比")
        print("=" * 60)
        run_npz_mode(model, args.npz, device, args.output)

    print("\n✓ Done!")
