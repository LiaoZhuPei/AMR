import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 配置区 ---

# 1. 设置您的 .iq 文件路径
IQ_FILE_PATH = 'E:/Project/AMR-Benchmark/QAM16-03-24-25-10h52m09s762.iq'

# 2. 设置您的 .iq 文件的数据格式 (例如 np.int16)
DATA_TYPE = np.int16

# 3. 滑动窗口参数
SEGMENT_LENGTH = 1024
STRIDE = 1024

# 4. 设置要丢弃的起始样本数
SAMPLES_TO_DISCARD = 10000

# 5. !!!!! 新增参数：设置要分析的最大窗口数量 !!!!!
#    - 设置为一个正整数 n, 例如 100, 表示只分析前 100 个窗口。
#    - 设置为 0 或 None, 表示分析整个文件。
MAX_SEGMENTS_TO_ANALYZE = 10  # <-- 在这里修改

# 6. 设置图像保存的文件夹
OUTPUT_DIR = 'iq_scan_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 主执行程序 ---

if __name__ == "__main__":
    if not os.path.exists(IQ_FILE_PATH):
        print(f"Error: IQ data file not found at '{IQ_FILE_PATH}'")
    else:
        try:
            # 1. 加载和预处理数据
            print(f"Loading data from '{IQ_FILE_PATH}'...")
            raw_data = np.fromfile(IQ_FILE_PATH, dtype=DATA_TYPE)
            iq_pairs = raw_data.reshape(-1, 2)

            if np.issubdtype(DATA_TYPE, np.integer):
                max_val = np.iinfo(DATA_TYPE).max
                iq_pairs = iq_pairs.astype(np.float32) / max_val

            print(f"Loaded {iq_pairs.shape[0]} total IQ samples.")

            # 2. 丢弃起始部分
            if iq_pairs.shape[0] > SAMPLES_TO_DISCARD:
                stable_iq_data = iq_pairs[SAMPLES_TO_DISCARD:]
                print(f"Discarded first {SAMPLES_TO_DISCARD} samples. {stable_iq_data.shape[0]} samples remaining.")
            else:
                stable_iq_data = iq_pairs

            # 3. 计算总共可以切分出多少个窗口
            total_possible_segments = (stable_iq_data.shape[0] - SEGMENT_LENGTH) // STRIDE + 1
            if total_possible_segments <= 0:
                raise ValueError("Not enough data to form a single segment after discarding.")

            # 4. !!!!! 根据新参数确定最终要分析的窗口数量 !!!!!
            if MAX_SEGMENTS_TO_ANALYZE and MAX_SEGMENTS_TO_ANALYZE > 0:
                num_segments_to_run = min(total_possible_segments, MAX_SEGMENTS_TO_ANALYZE)
                print(f"Analysis limited to the first {num_segments_to_run} segments.")
            else:
                num_segments_to_run = total_possible_segments
                print(f"Data will be scanned in {num_segments_to_run} segments (full scan).")

            # 5. 遍历并分析指定数量的窗口
            for i in tqdm(range(num_segments_to_run), desc="Analyzing Segments"):
                start_index = i * STRIDE
                end_index = start_index + SEGMENT_LENGTH
                iq_segment = stable_iq_data[start_index:end_index]
                iq_segment_transposed = iq_segment.T

                i_data = iq_segment_transposed[0, :]
                q_data = iq_segment_transposed[1, :]

                # --- 生成并保存图像 (这部分代码不变) ---
                plt.ioff()
                fig, axs = plt.subplots(1, 3, figsize=(20, 5))
                fig.suptitle(f'Analysis of IQ Segment | Start Sample: {start_index + SAMPLES_TO_DISCARD}', fontsize=16)

                axs[0].plot(i_data, label='I component')
                axs[0].plot(q_data, label='Q component')
                axs[0].set_title('I/Q Time Domain Waveform')
                axs[0].set_xlabel('Sample Index')
                axs[0].set_ylabel('Amplitude')
                axs[0].grid(True)
                axs[0].legend()

                axs[1].scatter(i_data, q_data, s=5)
                axs[1].set_title('Constellation Diagram')
                axs[1].set_xlabel('I (In-phase)')
                axs[1].set_ylabel('Q (Quadrature)')
                axs[1].grid(True)
                axs[1].set_aspect('equal', 'box')
                lim = np.max(np.abs(iq_segment_transposed)) * 1.2 + 1e-9
                axs[1].set_xlim(-lim, lim)
                axs[1].set_ylim(-lim, lim)

                complex_signal = i_data + 1j * q_data
                axs[2].psd(complex_signal, NFFT=SEGMENT_LENGTH, Fs=1)
                axs[2].set_title('Power Spectral Density (PSD)')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                filename = f'segment_start_{start_index + SAMPLES_TO_DISCARD:08d}.png'
                save_path = os.path.join(OUTPUT_DIR, filename)
                plt.savefig(save_path)
                plt.close(fig)

            print(f"\nAnalysis complete. {num_segments_to_run} plots saved in '{OUTPUT_DIR}' directory.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")