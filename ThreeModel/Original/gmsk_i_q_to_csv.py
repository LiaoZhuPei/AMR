import numpy as np
import os
from tqdm import tqdm

# --- 配置区（与生成 mat 的脚本保持一致）---
I_FILE_PATH = 'E:/Project/AMR-Benchmark/GMSK/I/ZPCJ_I_20260228_150905.dat'
Q_FILE_PATH = 'E:/Project/AMR-Benchmark/GMSK/Q/ZPCJ_Q_20260228_150905.dat'

DATA_TYPE = np.int16
SAMPLES_TO_DISCARD = 1000
SEGMENT_LENGTH = 1024     # 改为 1024
STRIDE = SEGMENT_LENGTH   # 无重叠

CSV_OUTPUT_DIR = 'output_csv_1024_GMSK5_verify'
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

# --- 主程序 ---
print("Step 1: Loading I/Q files...")
raw_i = np.fromfile(I_FILE_PATH, dtype=DATA_TYPE)
raw_q = np.fromfile(Q_FILE_PATH, dtype=DATA_TYPE)

min_len = min(len(raw_i), len(raw_q))
raw_i = raw_i[:min_len]
raw_q = raw_q[:min_len]

# 归一化（与 mat 生成脚本完全一致）
max_val = np.iinfo(DATA_TYPE).max
iq_pairs = np.stack((raw_i, raw_q), axis=1).astype(np.float32) / max_val

# 去除无效值
finite_mask = np.isfinite(iq_pairs).all(axis=1)
clean_iq_data = iq_pairs[finite_mask]

# 丢弃起始样本
stable_iq_data = clean_iq_data[SAMPLES_TO_DISCARD:]

num_segments = (stable_iq_data.shape[0] - SEGMENT_LENGTH) // STRIDE + 1
print(f"Data ready. 可切分段数: {num_segments}")

print("Step 2: Generating CSV files...")
for i in tqdm(range(num_segments), desc="Processing"):
    start_index = i * STRIDE
    end_index = start_index + SEGMENT_LENGTH

    segment_data = stable_iq_data[start_index:end_index]  # (1024, 2)
    segment_transposed = segment_data.T                    # (2, 1024)，第0行I，第1行Q

    i_row = segment_transposed[0]  # I路，1024个点
    q_row = segment_transposed[1]  # Q路，1024个点

    csv_filename = f'segment_{i:06d}.csv'
    csv_save_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)

    with open(csv_save_path, 'w') as f:
        # 第一行 I，第二行 Q，保留足够精度
        f.write(','.join([f'{x:.10f}' for x in i_row]) + '\n')
        f.write(','.join([f'{x:.10f}' for x in q_row]) + '\n')

print(f"\n完成！CSV 文件保存在: {CSV_OUTPUT_DIR}")
print(f"每个文件格式：第1行=I路(1024点)，第2行=Q路(1024点)")