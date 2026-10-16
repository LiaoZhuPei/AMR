import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm

# --- 配置区 ---

# 1. 输入文件路径 (I路和Q路分开)
# 请修改为你实际的本地路径
I_FILE_PATH = 'E:/Project/AMR-Benchmark/GMSK/I/ZPCJ_I_20260228_150909.dat'
Q_FILE_PATH = 'E:/Project/AMR-Benchmark/GMSK/Q/ZPCJ_Q_20260228_150909.dat'

# 2. 输入文件的数据格式
# 注意：如果不确定格式，可以先尝试 np.int16 (常见设备) 或 np.float32 (GNU Radio)
DATA_TYPE = np.int16

# 3. 数据处理参数
SAMPLES_TO_DISCARD = 1000  # 丢弃起始的不稳定样本
SEGMENT_LENGTH = 512      # 每个 .mat 文件的长度
STRIDE = SEGMENT_LENGTH    # 步长 (无重叠)

# 4. 输出配置
# 保存 .mat 文件的文件夹
MAT_OUTPUT_DIR = 'output_mat_512_GMSK9'
# 保存样本分析图的文件夹
PLOT_OUTPUT_DIR = 'output_plot_512_GMSK9'
# 要生成并保存分析图的样本数量
NUM_PLOTS_TO_SAVE = 20

# 5. .mat 文件中数据集的名称
MAT_DATASET_NAME = 'data_save'

# --- 创建输出文件夹 ---
os.makedirs(MAT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)


# --- 绘图函数 (保持不变) ---
def save_analysis_plot(iq_segment, segment_index):
    """为给定的IQ片段生成并保存分析图"""
    plt.ioff()
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    title = f'Analysis of Segment #{segment_index} (GMSK)'
    fig.suptitle(title, fontsize=16)

    # 准备数据，绘图函数输入形状为 (2, 1024)
    i_data = iq_segment[0, :]
    q_data = iq_segment[1, :]

    # 1. 时域波形
    axs[0].plot(i_data, label='I component', alpha=0.8)
    axs[0].plot(q_data, label='Q component', alpha=0.8)
    axs[0].set_title('I/Q Time Domain Waveform')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Amplitude')

    # 2. 星座图
    axs[1].scatter(i_data, q_data, s=5, alpha=0.6)
    axs[1].set_title('Constellation Diagram')
    axs[1].grid(True)
    axs[1].set_aspect('equal', 'box')
    axs[1].set_xlabel('I (In-phase)')
    axs[1].set_ylabel('Q (Quadrature)')
    # 动态设置坐标轴范围
    lim = np.max(np.abs(iq_segment)) * 1.2 + 1e-9
    axs[1].set_xlim(-lim, lim)
    axs[1].set_ylim(-lim, lim)

    # 3. 功率谱密度 (PSD)
    complex_signal = i_data + 1j * q_data
    axs[2].psd(complex_signal, NFFT=SEGMENT_LENGTH, Fs=1)
    axs[2].set_title('Power Spectral Density (PSD)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存文件
    filename = f'segment_{segment_index:06d}_report.png'
    save_path = os.path.join(PLOT_OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)


# --- 主执行程序 ---
if __name__ == "__main__":
    # 检查两个文件是否存在
    if not os.path.exists(I_FILE_PATH) or not os.path.exists(Q_FILE_PATH):
        print(f"Error: Files not found.\nCheck: {I_FILE_PATH}\nCheck: {Q_FILE_PATH}")
    else:
        try:
            # 1. 分别加载 I 和 Q 数据
            print(f"Step 1: Loading separate I and Q files...")
            raw_i = np.fromfile(I_FILE_PATH, dtype=DATA_TYPE)
            raw_q = np.fromfile(Q_FILE_PATH, dtype=DATA_TYPE)

            print(f"Loaded I points: {len(raw_i)}")
            print(f"Loaded Q points: {len(raw_q)}")

            # 2. 数据对齐与合并
            # 确保两路数据长度一致，取最小值
            min_len = min(len(raw_i), len(raw_q))
            if len(raw_i) != len(raw_q):
                print(f"Warning: File lengths mismatch. Truncating to {min_len} samples.")
            
            raw_i = raw_i[:min_len]
            raw_q = raw_q[:min_len]

            # 合并为 (N, 2) 的矩阵，第0列是I，第1列是Q
            # 这与参考代码中 reshape(-1, 2) 得到的结果结构一致
            iq_pairs = np.stack((raw_i, raw_q), axis=1)

            # 3. 归一化处理
            if np.issubdtype(DATA_TYPE, np.integer):
                max_val = np.iinfo(DATA_TYPE).max
                # 防止除零错误，虽然max通常不为0
                if max_val == 0: max_val = 1 
                iq_pairs = iq_pairs.astype(np.float32) / max_val
            
            # 检查无效值 (NaN/Inf)
            finite_mask = np.isfinite(iq_pairs).all(axis=1)
            clean_iq_data = iq_pairs[finite_mask]

            # 丢弃起始样本
            if clean_iq_data.shape[0] > SAMPLES_TO_DISCARD:
                stable_iq_data = clean_iq_data[SAMPLES_TO_DISCARD:]
            else:
                stable_iq_data = clean_iq_data

            print(f"Data ready. Usable IQ samples: {stable_iq_data.shape[0]}")

            # 4. 计算切分数量
            num_segments = (stable_iq_data.shape[0] - SEGMENT_LENGTH) // STRIDE + 1
            if num_segments <= 0:
                raise ValueError("Not enough data to form a single segment.")

            print(f"\nStep 2: Converting {num_segments} segments into .mat files...")

            # 5. 遍历切分并保存
            for i in tqdm(range(num_segments), desc="Processing"):
                start_index = i * STRIDE
                end_index = start_index + SEGMENT_LENGTH

                # 获取当前切片，形状 (1024, 2)
                segment_data = stable_iq_data[start_index:end_index]

                # --- 保存为 .mat 文件 ---
                # 转置为 (2, 1024) -> I在第一行，Q在第二行
                segment_transposed = segment_data.T

                # 增加 Batch 维度 -> (1, 2, 1024)
                segment_to_save = segment_transposed[np.newaxis, :, :]

                mat_filename = f'segment_{i:06d}.mat'
                mat_save_path = os.path.join(MAT_OUTPUT_DIR, mat_filename)

                with h5py.File(mat_save_path, 'w') as f:
                    f.create_dataset(MAT_DATASET_NAME, data=segment_to_save)

                # --- 生成分析图 ---
                if i < NUM_PLOTS_TO_SAVE:
                    save_analysis_plot(segment_transposed, i)

            print(f"\nProcessing complete!")
            print(f"Files saved to: {MAT_OUTPUT_DIR}")
            print(f"Output shape per file: {segment_to_save.shape}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")