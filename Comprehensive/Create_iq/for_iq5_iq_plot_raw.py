import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm

# --- 配置区 ---

# 1. 输入文件路径
IQ_FILE_PATH = 'E:/Project/AMR-Benchmark/ASK2-03-24-25-10h56m09s501.iq'

# 2. 输入文件的数据格式 (例如 np.int16)
DATA_TYPE = np.int16

# 3. 数据处理参数
SAMPLES_TO_DISCARD = 10000  # 丢弃的起始样本数
SEGMENT_LENGTH = 1024  # 每个 .mat 文件的长度
STRIDE = SEGMENT_LENGTH  # 步长，默认为 SEGMENT_LENGTH (无重叠)

# 4. 输出配置
MAT_OUTPUT_DIR = 'output_mat_files_ASK2_raw_amplitude'  # 使用新目录名以区分
PLOT_OUTPUT_DIR = 'output_plot_reports_ASK2_raw_amplitude'
NUM_PLOTS_TO_SAVE = 100

# 5. .mat 文件中数据集的名称
MAT_DATASET_NAME = 'data_save'

# --- 创建输出文件夹 ---
os.makedirs(MAT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)


# --- 绘图函数 (保持不变，因为它会自动适应幅度) ---
def save_iq_waveform_plot(iq_segment, segment_index):
    """为给定的IQ片段生成并保存一个简单的IQ时域波形图。"""
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6))

    i_data = iq_segment[0, :]
    q_data = iq_segment[1, :]
    time_axis = np.arange(len(i_data))

    ax.plot(time_axis, i_data, label='I-Channel', color='blue')
    ax.plot(time_axis, q_data, label='Q-Channel', color='red', alpha=0.8)

    base_filename = f"segment_{segment_index:06d}"
    ax.set_title(f"IQ Waveform - {base_filename}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Original Amplitude")  # Y轴标签更新
    ax.legend()
    ax.grid(True)

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
            print(f"Step 1: Loading data from '{IQ_FILE_PATH}'...")
            raw_data = np.fromfile(IQ_FILE_PATH, dtype=DATA_TYPE)
            iq_pairs = raw_data.reshape(-1, 2)

            # --- 【关键修正】: 彻底移除了幅值归一化步骤 ---
            # 仅仅将数据转换为浮点数类型，不进行任何除法或缩放操作
            iq_pairs_float = iq_pairs.astype(np.float32)
            print("Data conversion to float complete, original amplitude is preserved.")

            # 使用转换后的浮点数数据进行后续处理
            finite_mask = np.isfinite(iq_pairs_float).all(axis=1)
            clean_iq_data = iq_pairs_float[finite_mask]

            if clean_iq_data.shape[0] > SAMPLES_TO_DISCARD:
                stable_iq_data = clean_iq_data[SAMPLES_TO_DISCARD:]
            else:
                stable_iq_data = clean_iq_data

            print(f"Data cleaning complete. Usable IQ samples: {stable_iq_data.shape[0]}")

            # 2. 计算可切分的窗口总数
            num_segments = (stable_iq_data.shape[0] - SEGMENT_LENGTH) // STRIDE + 1
            if num_segments <= 0:
                raise ValueError("Not enough data to form a single segment after cleaning and discarding.")

            print(f"\nStep 2: Starting conversion of {num_segments} segments into .mat files...")

            # 3. 遍历、保存 .mat 文件并生成样本图
            segment_to_save = None
            for i in tqdm(range(num_segments), desc="Converting to .mat"):
                start_index = i * STRIDE
                end_index = start_index + SEGMENT_LENGTH
                segment_data = stable_iq_data[start_index:end_index]

                segment_transposed = segment_data.T
                segment_to_save = segment_transposed[np.newaxis, :, :]

                mat_filename = f'segment_{i:06d}.mat'
                mat_save_path = os.path.join(MAT_OUTPUT_DIR, mat_filename)

                with h5py.File(mat_save_path, 'w') as f:
                    f.create_dataset(MAT_DATASET_NAME, data=segment_to_save)

                if i < NUM_PLOTS_TO_SAVE:
                    save_iq_waveform_plot(segment_transposed, i)

            print(f"\nConversion complete!")
            print(f"{num_segments} .mat files saved in '{MAT_OUTPUT_DIR}'.")
            if segment_to_save is not None:
                print(f"Each .mat file contains data of shape {segment_to_save.shape}.")
            print(f"{min(num_segments, NUM_PLOTS_TO_SAVE)} plot reports saved in '{PLOT_OUTPUT_DIR}'.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")