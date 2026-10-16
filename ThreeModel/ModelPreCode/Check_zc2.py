import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

# ================= 1. 参数配置 =================
DATASET_PATH = './sionna_dataset/Sionna_head_mid1.npz'
NUM_SAMPLES_TO_CHECK = 500  # 检查前 10 个样本

# 信号与 ZC 参数
SPS = 4
HEAD_SYM = 192
L_HEAD_F = HEAD_SYM * SPS  # 768
U_ROOT = 25
RRC_TAPS = 101
ROLLOFF = 0.35


# ================= 2. 本地参考信号生成 =================
def zc_seq(num_sym, u=25):
    n = np.arange(num_sym)
    return np.exp(-1j * np.pi * u * n * (n + 1) / num_sym)


def design_rrc(ntaps, sps, beta):
    T = float(sps)
    n = np.arange(-(ntaps // 2), ntaps // 2 + 1, dtype=np.float64)
    h = np.zeros_like(n)
    for i, t in enumerate(n):
        if abs(t) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(t) - T / (4.0 * beta)) < 1e-12:
            h[i] = beta / np.sqrt(2) * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * t / T * (1 - beta)) + 4 * beta * t / T * np.cos(np.pi * t / T * (1 + beta)))
            den = np.pi * t / T * (1 - (4 * beta * t / T) ** 2)
            h[i] = num / den
    return (h / np.linalg.norm(h)).astype(np.float64)


def get_local_reference():
    zc = zc_seq(HEAD_SYM, U_ROOT)
    zc_up = np.zeros(HEAD_SYM * SPS, dtype=complex)
    zc_up[::SPS] = zc
    rrc_filt = design_rrc(RRC_TAPS, SPS, ROLLOFF)
    local_ref = np.convolve(zc_up, rrc_filt, mode='same')
    return local_ref / np.linalg.norm(local_ref)


# ================= 3. 核心：分段抗频偏互相关 =================
def robust_segmented_correlation(rx_signal, local_ref, num_segments=16):
    """
    分段非相干检测，完美抵抗强 CFO 带来的相位翻转
    """
    ref_len = len(local_ref)
    out_len = len(rx_signal) - ref_len + 1

    # 累加缓冲
    corr_sum = np.zeros(out_len)

    seg_len = ref_len // num_segments

    for i in range(num_segments):
        # 1. 切片提取一小段本地模板
        start_idx = i * seg_len
        end_idx = (i + 1) * seg_len
        ref_chunk = local_ref[start_idx:end_idx]

        # 2. 用这一小段去和整个长信号做互相关
        chunk_corr = signal.correlate(rx_signal, ref_chunk, mode='valid')

        # 3. 对齐延迟并取绝对值累加 (非相干积分)
        # 因为第 i 段模板本来就处在 i*seg_len 的位置，计算出来的峰也会往后偏移 i*seg_len
        aligned_corr = chunk_corr[start_idx: start_idx + out_len]
        corr_sum += np.abs(aligned_corr)

    return corr_sum


def detect_sync_header(rx_signal_complex, local_ref):
    """提取相关峰的位置"""
    # 采用 16 段分割抗频偏检测
    corr = robust_segmented_correlation(rx_signal_complex, local_ref, num_segments=16)

    # 因为分段累加，背景噪声的平均值也会上升，所以我们需要将均值剔除来设阈值
    mean_noise = np.mean(corr)
    peak_threshold = mean_noise + (np.max(corr) - mean_noise) * 0.5

    # 寻峰
    peaks, _ = signal.find_peaks(corr, height=peak_threshold, distance=L_HEAD_F // 2)

    return peaks, corr


# ================= 4. 主程序 =================
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"找不到数据集文件: {DATASET_PATH}")
        exit()

    print("加载数据集...")
    dataset = np.load(DATASET_PATH)

    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    SNR_test = dataset['snr_test']
    classes = dataset['classes']

    # 为了验证抗干扰效果，我们也可以看看低SNR的表现，这里选全部范围测试
    indices_to_test = np.arange(NUM_SAMPLES_TO_CHECK)
    local_ref = get_local_reference()

    print(f"\n物理参数 -> 长度: {L_HEAD_F}点 | ZC根: {U_ROOT} | 分段抗频偏: 开启 (16段)")
    print("-" * 75)
    print(f"{'样本':<6} | {'SNR':<5} | {'类别':<10} | {'检测个数':<8} | {'起始位置 (预期:0或1400/1600/1800/2000)'}")
    print("-" * 75)

    plt.figure(figsize=(15, 12))

    for i, idx in enumerate(indices_to_test):
        rx_signal_iq = X_test[idx]
        rx_signal_complex = rx_signal_iq[0, :] + 1j * rx_signal_iq[1, :]
        snr_val = SNR_test[idx]
        cls_name = classes[Y_test[idx]]

        # 执行检测
        peaks, corr = detect_sync_header(rx_signal_complex, local_ref)
        num_zc = len(peaks)
        peak_positions = ", ".join([str(p) for p in peaks]) if num_zc > 0 else "未检测到"

        print(f"[{idx:<4}] | {snr_val:>3.0f}dB | {cls_name:<10} | {num_zc:<8} | 🎯 {peak_positions}")

        # 画图前 8 个信号
        if i < 8:
            plt.subplot(8, 1, i + 1)
            plt.plot(corr, label='Segmented Correlation', color='#1f77b4')

            # 画峰值红线
            for p in peaks:
                plt.axvline(x=p, color='r', linestyle='-', alpha=0.9, linewidth=1.5)
                plt.text(p + 20, corr[p], f'P:{p}', color='r', fontweight='bold', va='center')

            # 画理论候选位置绿虚线
            for expected_pos in [0, 1400, 1600, 1800, 2000]:
                plt.axvline(x=expected_pos, color='g', linestyle=':', alpha=0.5)

            plt.title(f"Idx:{idx} | {cls_name} | SNR:{snr_val}dB", fontsize=10, pad=3)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 4800])
            plt.yticks([])  # 隐藏Y轴刻度使图面整洁

            if i == 7:
                plt.xlabel("Sample Index")

    plt.tight_layout()
    save_path = 'zc_robust_detection.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("-" * 75)
    print(f"\n✅ 检测完成！已生成抗频偏图表: {save_path}")