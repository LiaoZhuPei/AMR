import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# ============================================================
# 1. 严格加载数据
# ============================================================
print("正在加载测试集数据进行统计计算...")
DATA_ROOT = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataMatD1'

test_data_flat = pd.read_csv(f'{DATA_ROOT}/test/test.csv', header=None).values
test_raw = test_data_flat.reshape(-1, 2, 1024)

test_labels = pd.read_csv(f'{DATA_ROOT}/test/test_labels.csv', header=None).values.flatten()
test_snr = pd.read_csv(f'{DATA_ROOT}/test/test_snr.csv', header=None).values.flatten()

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']


# 你的归一化函数
def power_normalize(signal_data):
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data
    return signal_data / np.sqrt(power)


# ============================================================
# 2. 特征提取函数
# ============================================================
def calculate_statistics(target_class, target_snr=26):
    idx_class = classes.index(target_class)
    # 筛选出大于等于目标信噪比的所有该类数据
    mask = (test_labels == idx_class) & (test_snr >= target_snr)
    samples = test_raw[mask]

    if len(samples) == 0:
        return None

    all_A = []
    all_P = []
    paprs = []

    for sig in samples:
        sig_norm = power_normalize(sig)
        I, Q = sig_norm[0], sig_norm[1]

        A = np.sqrt(I ** 2 + Q ** 2)
        P = I ** 2 + Q ** 2

        all_A.append(A)
        all_P.append(P)

        # 计算单条序列的 PAPR (线性值，不是dB)
        papr = np.max(P) / (np.mean(P) + 1e-8)
        paprs.append(papr)

    # 展平所有序列以计算宏观统计量
    all_A = np.concatenate(all_A)

    stats = {
        'Class': target_class,
        'Samples': len(samples),
        'Mean_A': np.mean(all_A),
        'Std_A': np.std(all_A),
        'Max_A': np.max(all_A),
        'Avg_PAPR': np.mean(paprs),
        'Kurtosis_A': kurtosis(all_A)  # 峰度
    }
    return stats


# ============================================================
# 3. 执行计算并打印完美对齐的表格
# ============================================================
print(f"提取高信噪比 (SNR >= 26dB) 数据的统计学指纹...\n")

results = []
for qam in ['16QAM', '64QAM', '256QAM']:
    res = calculate_statistics(qam, 26)
    if res:
        results.append(res)

# 打印表头
header = f"{'类别':<10} | {'样本数':<8} | {'均值(Mean)':<12} | {'标准差(Std)':<12} | {'极限峰值(Max)':<14} | {'平均PAPR':<12} | {'峰度(Kurtosis)':<14}"
print("-" * len(header))
print(header)
print("-" * len(header))

# 打印数据
for r in results:
    row = (f"{r['Class']:<12}| {r['Samples']:<10}| {r['Mean_A']:<15.4f}| "
           f"{r['Std_A']:<15.4f}| {r['Max_A']:<17.4f}| {r['Avg_PAPR']:<14.4f}| {r['Kurtosis_A']:<14.4f}")
    print(row)
print("-" * len(header))