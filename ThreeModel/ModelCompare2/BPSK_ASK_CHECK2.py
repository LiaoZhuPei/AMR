"""
diagnose_bpsk_ask_v2.py
深入分析相位特征 —— BPSK和2ASK唯一可能的区别
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
snr = pd.read_csv(f'{DATA_ROOT}/test/test_snr.csv', header=None).values.flatten()

BPSK_IDX = 0
ASK_IDX = 7

# 只看高SNR样本（排除噪声干扰）
high_snr_mask = snr >= 10
bpsk_mask = (labels == BPSK_IDX) & high_snr_mask
ask_mask = (labels == ASK_IDX) & high_snr_mask

bpsk_data = data[bpsk_mask][:50]
ask_data = data[ask_mask][:50]

print(f"High SNR (>=10dB): BPSK={bpsk_mask.sum()}, 2ASK={ask_mask.sum()}")


def deep_analysis(samples, name):
    """深度分析：相位、瞬时频率、星座图"""
    I, Q = samples[:, 0, :], samples[:, 1, :]
    amp = np.sqrt(I ** 2 + Q ** 2 + 1e-10)
    phase = np.arctan2(Q, I)

    # 相位差分
    dp = np.diff(phase, axis=1)
    dp = np.arctan2(np.sin(dp), np.cos(dp))  # wrap to [-π, π]

    # 相位跳变直方图
    dp_flat = dp.flatten()

    print(f"\n{'=' * 50}")
    print(f"  {name} 深度分析 (高SNR)")
    print(f"{'=' * 50}")

    # 1. 相位差分统计
    print(f"  相位差分 mean: {dp_flat.mean():.6f}")
    print(f"  相位差分 std:  {dp_flat.std():.6f}")

    # 2. 大幅度相位跳变比例（接近π的跳变）
    big_jump_ratio = np.mean(np.abs(dp_flat) > 2.5)  # > ~143°
    print(f"  大相位跳变(>2.5rad)比例: {big_jump_ratio:.4f}")

    # 3. 小幅度相位变化比例
    small_change_ratio = np.mean(np.abs(dp_flat) < 0.3)  # < ~17°
    print(f"  小相位变化(<0.3rad)比例: {small_change_ratio:.4f}")

    # 4. I通道符号变化
    i_sign_change = np.mean(np.abs(np.diff(np.sign(I), axis=1)) > 0, axis=1).mean()
    print(f"  I通道符号翻转率: {i_sign_change:.4f}")

    # 5. 高幅度时的相位分布
    high_amp_mask = amp > amp.mean(axis=1, keepdims=True) * 0.8
    high_amp_phases = phase[high_amp_mask]
    print(f"  高幅度处相位std: {np.std(high_amp_phases):.4f}")

    # 6. 幅度自相关
    autocorr = []
    for s in samples[:20]:
        a = np.sqrt(s[0] ** 2 + s[1] ** 2)
        a = a - a.mean()
        ac = np.correlate(a, a, 'full')
        ac = ac[len(ac) // 2:]
        ac = ac / (ac[0] + 1e-10)
        autocorr.append(ac[:100])
    autocorr = np.array(autocorr).mean(0)
    print(f"  幅度自相关[10]: {autocorr[10]:.4f}")
    print(f"  幅度自相关[20]: {autocorr[20]:.4f}")
    print(f"  幅度自相关[50]: {autocorr[50]:.4f}")

    # 7. I*Q 统计
    iq_product = (I * Q).mean(axis=1).mean()
    print(f"  mean(I*Q): {iq_product:.6f}")

    # 8. |I| vs |Q| 比较
    i_abs_mean = np.abs(I).mean()
    q_abs_mean = np.abs(Q).mean()
    print(f"  mean(|I|): {i_abs_mean:.6f}")
    print(f"  mean(|Q|): {q_abs_mean:.6f}")
    print(f"  |I|/|Q| ratio: {i_abs_mean / (q_abs_mean + 1e-10):.4f}")

    return dp_flat, phase, amp, autocorr


bpsk_dp, bpsk_ph, bpsk_amp, bpsk_ac = deep_analysis(bpsk_data, "BPSK")
ask_dp, ask_ph, ask_amp, ask_ac = deep_analysis(ask_data, "2ASK")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('BPSK vs 2ASK Deep Diagnostic (High SNR)', fontsize=14)

# Row 1: 相位差分直方图
axes[0, 0].hist(bpsk_dp, bins=200, density=True, color='blue', alpha=0.7)
axes[0, 0].set_title('BPSK - Phase Diff Histogram')
axes[0, 0].set_xlabel('Phase difference (rad)')
axes[0, 0].axvline(x=np.pi, color='r', linestyle='--', label='π')
axes[0, 0].axvline(x=-np.pi, color='r', linestyle='--')
axes[0, 0].legend()

axes[0, 1].hist(ask_dp, bins=200, density=True, color='red', alpha=0.7)
axes[0, 1].set_title('2ASK - Phase Diff Histogram')
axes[0, 1].set_xlabel('Phase difference (rad)')
axes[0, 1].axvline(x=np.pi, color='r', linestyle='--', label='π')
axes[0, 1].axvline(x=-np.pi, color='r', linestyle='--')
axes[0, 1].legend()

# Row 2: 星座图
idx = 0
axes[1, 0].scatter(bpsk_data[idx, 0, :200], bpsk_data[idx, 1, :200],
                   s=2, alpha=0.5, c='blue')
axes[1, 0].set_title('BPSK - Constellation (1 sample)')
axes[1, 0].set_xlabel('I')
axes[1, 0].set_ylabel('Q')
axes[1, 0].set_aspect('equal')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(ask_data[idx, 0, :200], ask_data[idx, 1, :200],
                   s=2, alpha=0.5, c='red')
axes[1, 1].set_title('2ASK - Constellation (1 sample)')
axes[1, 1].set_xlabel('I')
axes[1, 1].set_ylabel('Q')
axes[1, 1].set_aspect('equal')
axes[1, 1].grid(True, alpha=0.3)

# Row 3: 幅度自相关
axes[2, 0].plot(bpsk_ac, 'b-', label='BPSK')
axes[2, 0].plot(ask_ac, 'r-', label='2ASK')
axes[2, 0].set_title('Amplitude Autocorrelation Comparison')
axes[2, 0].set_xlabel('Lag')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Row 3 right: I通道波形对比
t = np.arange(200)
axes[2, 1].plot(t, bpsk_data[0, 0, :200], 'b-', linewidth=0.5, label='BPSK', alpha=0.7)
axes[2, 1].plot(t, ask_data[0, 0, :200], 'r-', linewidth=0.5, label='2ASK', alpha=0.7)
axes[2, 1].set_title('I Channel Waveform Comparison')
axes[2, 1].set_xlabel('Sample')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure/bpsk_vs_ask_deep.png', dpi=150)
plt.show()

# ============================================================
# 最终判断
# ============================================================
print("\n" + "=" * 60)
print("最终判断")
print("=" * 60)

bpsk_big_jump = np.mean(np.abs(bpsk_dp) > 2.5)
ask_big_jump = np.mean(np.abs(ask_dp) > 2.5)
jump_diff = abs(bpsk_big_jump - ask_big_jump)

bpsk_amp_std = bpsk_amp.std(axis=1).mean() / bpsk_amp.mean(axis=1).mean()
ask_amp_std = ask_amp.std(axis=1).mean() / ask_amp.mean(axis=1).mean()
amp_diff = abs(bpsk_amp_std - ask_amp_std)

print(f"  大相位跳变率差异: {jump_diff:.4f}  ", end="")
print("← 有区别 ✓" if jump_diff > 0.02 else "← 几乎相同 ✗")
print(f"  幅度变异系数差异: {amp_diff:.4f}  ", end="")
print("← 有区别 ✓" if amp_diff > 0.02 else "← 几乎相同 ✗")

if jump_diff < 0.02 and amp_diff < 0.02:
    print("\n  ⚠️ 结论: BPSK 和 2ASK 在该数据集中物理特征完全一致")
    print("  → 很可能是脉冲成形后的双极性2ASK = BPSK")
    print("  → 模型无法区分是正常的，建议合并为同一类")
elif jump_diff > 0.02:
    print("\n  ✓ 相位特征可区分，在 stat_features 中增加相位跳变直方图")
elif amp_diff > 0.02:
    print("\n  ✓ 幅度特征可区分，当前统计特征应该够用")
    print("  → 问题可能在于相位旋转增强破坏了差异")
