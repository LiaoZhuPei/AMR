"""
diagnose_bpsk_ask.py
独立运行，深度分析 BPSK 和 2ASK 在数据集中的真实差异
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

DATA_ROOT = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataMatD1'
SIGNAL_LENGTH = 1024

data = pd.read_csv(f'{DATA_ROOT}/test/test.csv', header=None).values.reshape(-1, 2, SIGNAL_LENGTH)
labels = pd.read_csv(f'{DATA_ROOT}/test/test_labels.csv', header=None).values.flatten()

BPSK_IDX = classes.index('BPSK')   # 0
ASK_IDX = classes.index('2ASK')    # 7

bpsk_data = data[labels == BPSK_IDX]
ask_data = data[labels == ASK_IDX]

print(f"BPSK samples (label={BPSK_IDX}): {len(bpsk_data)}")
print(f"2ASK samples (label={ASK_IDX}): {len(ask_data)}")


def analyze_signals(samples, name, n=50):
    """分析信号的物理特征"""
    samples = samples[:n]
    I, Q = samples[:, 0, :], samples[:, 1, :]
    amp = np.sqrt(I ** 2 + Q ** 2)

    print(f"\n{'=' * 50}")
    print(f"  {name} 原始信号分析 (未归一化)")
    print(f"{'=' * 50}")

    # 1. I/Q 均值（BPSK应该接近0，OOK可能有偏移）
    i_mean = I.mean(axis=1).mean()
    q_mean = Q.mean(axis=1).mean()
    print(f"  I均值: {i_mean:.6f}")
    print(f"  Q均值: {q_mean:.6f}")

    # 2. 幅度统计
    amp_mean = amp.mean()
    amp_std_within = amp.std(axis=1).mean()
    amp_min = amp.min(axis=1).mean()
    amp_max = amp.max(axis=1).mean()
    print(f"  幅度均值: {amp_mean:.6f}")
    print(f"  幅度std(样本内): {amp_std_within:.6f}")
    print(f"  幅度最小值(均): {amp_min:.6f}")
    print(f"  幅度最大值(均): {amp_max:.6f}")
    print(f"  幅度最小/最大比: {amp_min / (amp_max + 1e-10):.6f}")

    # 3. I通道值的分布（最关键）
    i_values = I.flatten()
    print(f"  I通道 std: {i_values.std():.6f}")
    print(f"  I通道 min: {i_values.min():.6f}")
    print(f"  I通道 max: {i_values.max():.6f}")

    # 4. 功率归一化后的统计
    power = np.mean(I ** 2 + Q ** 2, axis=1, keepdims=True)
    I_norm = I / np.sqrt(power + 1e-10)
    Q_norm = Q / np.sqrt(power + 1e-10)
    amp_norm = np.sqrt(I_norm ** 2 + Q_norm ** 2)

    amp_norm_std = amp_norm.std(axis=1).mean()
    amp_norm_min = amp_norm.min(axis=1).mean()
    print(f"\n  [功率归一化后]")
    print(f"  幅度std(样本内): {amp_norm_std:.6f}")
    print(f"  幅度最小值(均): {amp_norm_min:.6f}")

    # 判断
    if amp_norm_std < 0.05:
        print(f"  → 恒定幅度信号 (像PSK/FSK)")
    elif amp_norm_min < 0.1:
        print(f"  → 有零值/近零值 (像OOK)")
    else:
        print(f"  → 多级幅度信号 (像ASK/QAM)")

    return I, Q, amp, I_norm, Q_norm, amp_norm


bpsk_I, bpsk_Q, bpsk_amp, bpsk_In, bpsk_Qn, bpsk_ampn = analyze_signals(bpsk_data, "BPSK")
ask_I, ask_Q, ask_amp, ask_In, ask_Qn, ask_ampn = analyze_signals(ask_data, "2ASK")

# ============================================================
# 可视化对比
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 14))
fig.suptitle('BPSK vs 2ASK Diagnostic', fontsize=14)

sample_idx = 0
t = np.arange(200)  # 前200个点

# Row 1: 原始I通道波形
axes[0, 0].plot(t, bpsk_data[sample_idx, 0, :200], 'b-', linewidth=0.5)
axes[0, 0].set_title(f'BPSK (label={BPSK_IDX}) - I channel (raw)')
axes[0, 0].set_ylabel('Amplitude')

axes[0, 1].plot(t, ask_data[sample_idx, 0, :200], 'r-', linewidth=0.5)
axes[0, 1].set_title(f'2ASK (label={ASK_IDX}) - I channel (raw)')

# Row 2: 幅度波形
axes[1, 0].plot(t, bpsk_amp[sample_idx, :200], 'b-', linewidth=0.5)
axes[1, 0].set_title('BPSK - Amplitude (raw)')
axes[1, 0].set_ylabel('|s(t)|')

axes[1, 1].plot(t, ask_amp[sample_idx, :200], 'r-', linewidth=0.5)
axes[1, 1].set_title('2ASK - Amplitude (raw)')

# Row 3: 归一化后幅度
axes[2, 0].plot(t, bpsk_ampn[sample_idx, :200], 'b-', linewidth=0.5)
axes[2, 0].set_title('BPSK - Amplitude (power normalized)')
axes[2, 0].set_ylabel('|s(t)| norm')

axes[2, 1].plot(t, ask_ampn[sample_idx, :200], 'r-', linewidth=0.5)
axes[2, 1].set_title('2ASK - Amplitude (power normalized)')

# Row 4: I通道值直方图
axes[3, 0].hist(bpsk_data[:10, 0, :].flatten(), bins=100, color='blue', alpha=0.7)
axes[3, 0].set_title('BPSK - I channel histogram')
axes[3, 0].set_xlabel('I value')

axes[3, 1].hist(ask_data[:10, 0, :].flatten(), bins=100, color='red', alpha=0.7)
axes[3, 1].set_title('2ASK - I channel histogram')
axes[3, 1].set_xlabel('I value')

plt.tight_layout()
plt.savefig('figure/bpsk_vs_ask_diagnostic.png', dpi=150)
plt.show()

# ============================================================
# 关键结论
# ============================================================
print("\n" + "=" * 60)
print("关键比较")
print("=" * 60)

bpsk_amp_cv = bpsk_ampn.std(axis=1).mean() / bpsk_ampn.mean(axis=1).mean()
ask_amp_cv = ask_ampn.std(axis=1).mean() / ask_ampn.mean(axis=1).mean()
print(f"  BPSK 幅度变异系数(归一化后): {bpsk_amp_cv:.6f}")
print(f"  2ASK 幅度变异系数(归一化后): {ask_amp_cv:.6f}")
print(f"  两者差异: {abs(bpsk_amp_cv - ask_amp_cv):.6f}")

if abs(bpsk_amp_cv - ask_amp_cv) < 0.05:
    print("\n  ⚠️ 警告: 归一化后 BPSK 和 2ASK 的幅度特征几乎相同！")
    print("  → 功率归一化可能抹除了两者的区别")
    print("  → 需要修改归一化策略或增加区分特征")
else:
    print("\n  ✓ 归一化后两者仍有明显区别")
    print("  → 可能是模型训练问题或标签问题")
print("=" * 60)