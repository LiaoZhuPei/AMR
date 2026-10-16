import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 配置区 ---

# 1. 输入文件路径
IQ_FILE_PATH = 'E:/Project/AMR-Benchmark/DDC03_64QAM_3_125M.dat'

# 2. 输入文件的数据格式 (例如 np.int16)
DATA_TYPE = np.int16

# 3. 数据处理参数
SAMPLES_TO_DISCARD = 10000  # 丢弃的起始样本数
SEGMENT_LENGTH = 1024  # 每个片段的样本长度
#SEGMENT_LENGTH = 500  # 每个片段的样本长度
# 步长，默认为 SEGMENT_LENGTH (无重叠)。
STRIDE = SEGMENT_LENGTH

# 4. 输出配置
# 保存 .csv 文件的文件夹
CSV_OUTPUT_DIR = 'output_csv_files_1024_QAM64_DDC'
#CSV_OUTPUT_DIR = 'output_csv_files_256QAM'
# 保存样本分析图的文件夹
PLOT_OUTPUT_DIR = 'output_plot_reports_1024_QAM64_DDC'
#PLOT_OUTPUT_DIR = 'output_plot_reports_256QAM'
# 要生成并保存分析图的样本数量
NUM_PLOTS_TO_SAVE = 5

# --- 创建输出文件夹 ---
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)


# --- 绘图函数 ---
def save_analysis_plot(iq_segment_transposed, segment_index, original_start_sample):
    """
    为给定的IQ片段生成并保存分析图。
    输入 iq_segment_transposed 的形状应为 (2, 1024)。
    """
    plt.ioff()  # 关闭交互模式
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    title = f'Analysis of Segment #{segment_index} | Original Start Sample: {original_start_sample}'
    fig.suptitle(title, fontsize=16)

    # 准备数据，绘图函数输入形状已为 (2, 1024)
    i_data = iq_segment_transposed[0, :]
    q_data = iq_segment_transposed[1, :]

    # 绘图逻辑
    axs[0].plot(i_data, label='I component');
    axs[0].plot(q_data, label='Q component')
    axs[0].set_title('I/Q Time Domain Waveform');
    axs[0].grid(True);
    axs[0].legend()
    axs[0].set_xlabel('Sample Index');
    axs[0].set_ylabel('Amplitude')

    axs[1].scatter(i_data, q_data, s=5);
    axs[1].set_title('Constellation Diagram')
    axs[1].grid(True);
    axs[1].set_aspect('equal', 'box');
    axs[1].set_xlabel('I (In-phase)');
    axs[1].set_ylabel('Q (Quadrature)')
    lim = np.max(np.abs(iq_segment_transposed)) * 1.2 + 1e-9
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

            if np.issubdtype(DATA_TYPE, np.integer):
                max_val = np.iinfo(DATA_TYPE).max
                iq_pairs = iq_pairs.astype(np.float32) / max_val

            finite_mask = np.isfinite(iq_pairs).all(axis=1)
            clean_iq_data = iq_pairs[finite_mask]

            if clean_iq_data.shape[0] > SAMPLES_TO_DISCARD:
                stable_iq_data = clean_iq_data[SAMPLES_TO_DISCARD:]
            else:
                stable_iq_data = clean_iq_data

            print(f"Data cleaning complete. Usable IQ samples: {stable_iq_data.shape[0]}")

            # 2. 计算可切分的窗口总数
            num_segments = (stable_iq_data.shape[0] - SEGMENT_LENGTH) // STRIDE + 1
            if num_segments <= 0:
                raise ValueError("Not enough data to form a single segment after cleaning and discarding.")

            print(f"\nStep 2: Starting conversion of {num_segments} segments into .csv files...")

            # 3. 遍历、保存 .csv 文件并生成样本图
            for i in tqdm(range(num_segments), desc="Converting to .csv"):
                start_index = i * STRIDE
                end_index = start_index + SEGMENT_LENGTH

                segment_data = stable_iq_data[start_index:end_index]  # Shape: (1024, 2)

                # --- 关键修改：转置数据以满足 (2, 1024) 的格式要求 ---
                segment_to_save = segment_data.T  # Shape: (2, 1024)

                # --- 保存为 .csv 文件 ---
                csv_filename = f'segment_{i:06d}.csv'
                csv_save_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)

                # 直接保存 (2, 1024) 的数组。np.savetxt 会创建两行，每行1024个值。
                # 对于这种格式，添加表头没有意义，因此移除 header 参数。
                np.savetxt(csv_save_path, segment_to_save, delimiter=',')

                # --- 为前 N 个样本生成并保存分析图 ---
                if i < NUM_PLOTS_TO_SAVE:
                    # 绘图函数也需要 (2, 1024) 的数据，我们直接使用转置后的 segment_to_save
                    try:
                        original_start_sample_pos = np.where(iq_pairs == segment_data[0])[0][0]
                    except IndexError:
                        original_start_sample_pos = -1  # 如果找不到，则标记为-1

                    save_analysis_plot(segment_to_save, i, original_start_sample_pos)

            print(f"\nConversion complete!")
            print(f"{num_segments} .csv files saved in '{CSV_OUTPUT_DIR}'.")
            print(
                f"Each .csv file contains data of shape {segment_to_save.shape} (2 rows: I and Q, {SEGMENT_LENGTH} columns: samples).")
            print(f"{min(num_segments, NUM_PLOTS_TO_SAVE)} plot reports saved in '{PLOT_OUTPUT_DIR}'.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")