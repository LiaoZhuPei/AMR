"""
quick_check.py — 独立运行，验证数据集标签是否正确
"""
import numpy as np
import pandas as pd

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

DATA_ROOT = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataMatD1'
SIGNAL_LENGTH = 1024

# 加载
data = pd.read_csv(f'{DATA_ROOT}/test/test.csv', header=None).values
data = data.reshape(-1, 2, SIGNAL_LENGTH)
labels = pd.read_csv(f'{DATA_ROOT}/test/test_labels.csv', header=None).values.flatten()

print("=" * 70)
print("每个标签的信号物理特征分析")
print("=" * 70)
print(f"{'Index':>5s}  {'Class':>12s}  {'Count':>6s}  "
      f"{'AmpMean':>8s}  {'AmpStd':>8s}  {'PhaseStd':>9s}  {'判断'}")
print("-" * 70)

for idx in sorted(set(labels)):
    mask = (labels == idx)
    samples = data[mask][:50]  # 取50个样本

    I, Q = samples[:, 0, :], samples[:, 1, :]
    amp = np.sqrt(I ** 2 + Q ** 2)
    phase = np.arctan2(Q, I)

    # 每个样本内部的幅度标准差（归一化后）
    amp_normed = amp / (amp.mean(axis=1, keepdims=True) + 1e-10)
    amp_std = amp_normed.std(axis=1).mean()

    # 相位变化
    dp = np.diff(phase, axis=1)
    dp = np.arctan2(np.sin(dp), np.cos(dp))
    phase_std = dp.std(axis=1).mean()

    amp_mean = amp.mean()

    # 自动判断
    if amp_std < 0.15 and phase_std > 0.5:
        guess = "→ PSK类"
    elif amp_std > 0.3 and phase_std < 0.5:
        guess = "→ ASK类"
    elif amp_std < 0.15 and phase_std < 0.3:
        guess = "→ 恒包络(FSK/MSK)"
    else:
        guess = "→ QAM/混合"

    cls_name = classes[idx] if idx < len(classes) else f"UNK_{idx}"
    print(f"{idx:>5d}  {cls_name:>12s}  {mask.sum():>6d}  "
          f"{amp_mean:>8.4f}  {amp_std:>8.4f}  {phase_std:>9.4f}  {guess}")

print("=" * 70)
print()
print("关键检查:")
print("  如果 label=0(BPSK) 的 AmpStd 大且显示 '→ ASK类'")
print("  而 label=7(2ASK) 的 AmpStd 小且显示 '→ PSK类'")
print("  → 说明 classes 列表和数据标签的映射是反的！")
print()
print("  修复方法: 调整 classes 列表的顺序，使其匹配数据实际含义")