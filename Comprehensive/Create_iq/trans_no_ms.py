import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm

# --- 配置区 ---

# 1. 输入文件路径
IQ_FILE_PATH = 'E:/Project/AMR-Benchmark/PI4DQPSK-03-24-25-10h54m23s982.iq'

# 2. 输入文件的数据格式 (例如 np.int16)
DATA_TYPE = np.int16

# 3. 数据处理参数
SAMPLES_TO_DISCARD = 10000  # 丢弃的起始样本数
SEGMENT_LENGTH = 1024  # 每个 .mat 文件的长度
# 步长，默认为 SEGMENT_LENGTH (无重叠)。
STRIDE = SEGMENT_LENGTH

# 4. 输出配置
# 保存 .mat 文件的文件夹
MAT_OUTPUT_DIR = 'output_mat_files_PI4DQPSK_raw_amplitude'
# 保存样本分析图的文件夹
PLOT_OUTPUT_DIR = 'output_plot_reports_PI4DQPSK_raw_amplitude'
# 要生成并保存分析图的样本数量
NUM_PLOTS_TO_SAVE = 100

# 5. .mat 文件中数据集的名称
MAT_DATASET_NAME = 'data_save'

# --- 创建输出文件夹 ---
os.makedirs(MAT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)


# --- 绘图函数 ---
def save_analysis_plot(iq_segment, segment_index, original_start_sample):
    """为给定的IQ片段生成并保存分析图"""
    plt.ioff()  # 关闭交互模式
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    title = f'Analysis of Segment #{segment_index} | Original Start Sample: {original_start_sample}'
    fig.suptitle(title, fontsize=16)

    # 准备数据，绘图函数输入形状为 (2, 1024)
    i_data = iq_segment[0, :]
    q_data = iq_segment[1, :]

    # 绘图逻辑
    axs[0].plot(i_data, label='I component');
    axs[0].plot(q_data, label='Q component')
    axs[0].set_title('I/Q Time Domain Waveform');
    axs[0].grid(True);
    axs[0].legend()
    axs[0].set_xlabel('Sample Index');
    axs[0].set_ylabel('Amplitude (Raw Integer Value)') # 坐标轴标签已更新

    axs[1].scatter(i_data, q_data, s=5);
    axs[1].set_title('Constellation Diagram')
    axs[1].grid(True);
    axs[1].set_aspect('equal', 'box');
    axs[1].set_xlabel('I (In-phase)');
    axs[1].set_ylabel('Q (Quadrature)')
    # 限制坐标轴以更好地显示
    lim = np.max(np.abs(iq_segment)) * 1.2 + 1e-9
    axs[1].set_xlim(-lim, lim);
    axs[1].set_ylim(-lim, lim)

    complex_signal = i_data + 1j * q_data
    axs[2].psd(complex_signal, NFFT=SEGMENT_LENGTH, Fs=1);
    axs[2].set_title('Power Spectral Density (PSD)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存文件
    filename = f'segment_{segment_index:06d}_report.png'
    save_path = os.path.join(PLOT_OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)


# --- 主执行程序 ---
if __name__ == "__main__":
    if not os.path.exists(IQ_FILE_PATH):
        print(f"Error: IQ data file not found at '{IQ_FILE_PATH}'")
    else:
        try:
            # 1. 加载和清洗数据
            print(f"Step 1: Loading and cleaning data from '{IQ_FILE_PATH}'...")
            raw_data = np.fromfile(IQ_FILE_PATH, dtype=DATA_TYPE)
            iq_pairs = raw_data.reshape(-1, 2)

            # --- !!!!!!!!!!!!! KEY CHANGE !!!!!!!!!!!!! ---
            # 原来的归一化代码块已被移除。
            # 我们只将数据类型转换为 float32，以保持与后续处理的兼容性，
            # 但不改变其原始幅值。
            iq_pairs = iq_pairs.astype(np.float32)
            # --- !!!!!!!!!!!!! END OF CHANGE !!!!!!!!!!!!! ---

            finite_mask = np.isfinite(iq_pairs).all(axis=1)
            clean_iq_data = iq_pairs[finite_mask]

            if clean_iq_data.shape[0] > SAMPLES_TO_DISCARD:
                stable_iq_data = clean_iq_data[SAMPLES_TO_DISCARD:]
            else:
                stable_iq_data = clean_iq_data

            print(f"Data cleaning complete. Usable IQ samples: {stable_iq_data.shape[0]}")
            # 打印一下现在数据的范围以供验证
            print(f"Sample data range after loading: min={np.min(stable_iq_data)}, max={np.max(stable_iq_data)}")


            # 2. 计算可切分的窗口总数
            num_segments = (stable_iq_data.shape[0] - SEGMENT_LENGTH) // STRIDE + 1
            if num_segments <= 0:
                raise ValueError("Not enough data to form a single segment after cleaning and discarding.")

            print(f"\nStep 2: Starting conversion of {num_segments} segments into .mat files...")

            # 3. 遍历、保存 .mat 文件并生成样本图
            for i in tqdm(range(num_segments), desc="Converting to .mat"):
                start_index = i * STRIDE
                end_index = start_index + SEGMENT_LENGTH

                segment_data = stable_iq_data[start_index:end_index]  # Shape: (1024, 2)

                # --- 保存为 .mat 文件 ---
                segment_transposed = segment_data.T # Shape: (2, 1024)
                segment_to_save = segment_transposed[np.newaxis, :, :] # Shape: (1, 2, 1024)

                mat_filename = f'segment_{i:06d}.mat'
                mat_save_path = os.path.join(MAT_OUTPUT_DIR, mat_filename)

                with h5py.File(mat_save_path, 'w') as f:
                    f.create_dataset(MAT_DATASET_NAME, data=segment_to_save)

                # --- 为前 N 个样本生成并保存分析图 ---
                if i < NUM_PLOTS_TO_SAVE:
                    original_start_sample_pos = np.where(iq_pairs == segment_data[0])[0][0]
                    save_analysis_plot(segment_transposed, i, original_start_sample_pos)

            print(f"\nConversion complete!")
            print(f"{num_segments} .mat files saved in '{MAT_OUTPUT_DIR}'.")
            print(f"Each .mat file contains data of shape {segment_to_save.shape}.")
            print(f"{min(num_segments, NUM_PLOTS_TO_SAVE)} plot reports saved in '{PLOT_OUTPUT_DIR}'.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")