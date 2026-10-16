import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os

# ================= 1. 参数配置 =================
DATASET_PATH = './sionna_dataset/Sionna_head_mid1.npz'
NUM_SAMPLES_TO_CHECK = 100  # 检测前几个样本

# 信号与 ZC 参数 (必须与生成脚本完全一致)
SPS = 4
HEAD_SYM = 192
L_HEAD_F = HEAD_SYM * SPS  # 768
U_ROOT = 25
RRC_TAPS = 101
ROLLOFF = 0.35


# ================= 2. 本地参考信号生成 =================
def zc_seq(num_sym, u=25):
    """生成根 ZC 序列"""
    n = np.arange(num_sym)
    return np.exp(-1j * np.pi * u * n * (n + 1) / num_sym)


def design_rrc(ntaps, sps, beta):
    """生成 RRC 滤波器"""
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


# 构建完美的本地参考 ZC 序列 (含上采样和RRC滤波)
def get_local_reference():
    zc = zc_seq(HEAD_SYM, U_ROOT)
    zc_up = np.zeros(HEAD_SYM * SPS, dtype=complex)
    zc_up[::SPS] = zc
    rrc_filt = design_rrc(RRC_TAPS, SPS, ROLLOFF)
    # 用 same 模式卷积，保持中心对齐
    local_ref = np.convolve(zc_up, rrc_filt, mode='same')
    # 归一化参考信号能量
    return local_ref / np.linalg.norm(local_ref)


# ================= 3. 检测核心函数 =================
def detect_sync_header(rx_signal_complex, local_ref):
    """
    通过互相关检测 ZC 序列的位置和个数
    """
    # 1. 计算滑动互相关的绝对值 (抵抗相位旋转)
    corr = np.abs(signal.correlate(rx_signal_complex, local_ref, mode='valid'))

    # 2. 寻找峰值 (设定阈值为最大峰值的 60%，以防假阳性)
    peak_threshold = np.max(corr) * 0.6
    peaks, _ = signal.find_peaks(corr, height=peak_threshold, distance=L_HEAD_F // 2)

    return peaks, corr


# ================= 4. 主程序 =================
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"找不到数据集文件: {DATASET_PATH}")
        exit()

    print("加载数据集...")
    dataset = np.load(DATASET_PATH)

    # 提取测试集，并为了看清相关峰，筛选高 SNR 样本进行展示 (比如 SNR > 10dB)
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    SNR_test = dataset['snr_test']
    classes = dataset['classes']

    high_snr_indices = np.where(SNR_test >= 20)[0]
    indices_to_test = high_snr_indices[:NUM_SAMPLES_TO_CHECK]

    local_ref = get_local_reference()
    print(f"\n系统设定同步头物理长度: {L_HEAD_F} 采样点 (192符号 x 4SPS)")
    print("-" * 75)
    print(f"{'样本索引':<10} | {'信噪比':<10} | {'类别':<12} | {'检测到个数':<10} | {'起始位置 (采样点)'}")
    print("-" * 75)

    # 准备画图
    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(indices_to_test):
        # 将 (2, 4800) 的 numpy 数组转回复数信号 I + jQ
        rx_signal_iq = X_test[idx]
        rx_signal_complex = rx_signal_iq[0, :] + 1j * rx_signal_iq[1, :]
        snr_val = SNR_test[idx]
        cls_name = classes[Y_test[idx]]

        # 检测 ZC 序列
        peaks, corr = detect_sync_header(rx_signal_complex, local_ref)

        num_zc = len(peaks)
        peak_positions = ", ".join([str(p) for p in peaks])

        print(f"[{idx:<8}] | {snr_val:>4.0f} dB   | {cls_name:<12} | {num_zc:<10} | {peak_positions}")

        # 画出前 5 个样本的互相关图
        if i < 100:
            plt.subplot(100, 1, i + 1)
            plt.plot(corr, label='Cross-Correlation Magnitude', color='b')
            for p in peaks:
                plt.axvline(x=p, color='r', linestyle='--', alpha=0.7)
                plt.text(p, corr[p], f' Peak @ {p}', color='r', fontweight='bold', va='bottom')

            # 由于数据集里模式可能是首部(0)或中部(1400,1600,1800,2000)，画辅助线标记理论可能的位置
            for expected_pos in [0, 1400, 1600, 1800, 2000]:
                plt.axvline(x=expected_pos, color='g', linestyle=':', alpha=0.3)

            plt.title(f"Sample {idx} - {cls_name} (SNR: {snr_val}dB) - Peak count: {num_zc}")
            plt.ylabel("Magnitude")
            plt.grid(True, alpha=0.3)

            if i == 4:
                plt.xlabel("Sample Index (Start Position)")

    plt.tight_layout()
    save_path = 'zc_correlation_peaks.png'
    plt.savefig(save_path, dpi=150)
    print("-" * 75)
    print(f"\n已生成相关峰可视化图: {save_path} (绿线为理论生成池中的候选位置，红线为实际检测到的位置)")