import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- 配置区 ---

# ===== 改动 1：输入改为 I 路和 Q 路两个独立的 .dat 文件 =====
I_FILE_PATH = 'E:/Project/AMR-Benchmark/GMSK/I/ZPCJ_I_20260228_150911.dat'
Q_FILE_PATH = 'E:/Project/AMR-Benchmark/GMSK/Q/ZPCJ_Q_20260228_150911.dat'

# ===== 改动 2：数据格式（根据实际 .dat 文件格式选择） =====
# 常见格式: np.float32, np.float64, np.int16, np.int32
# 如果不确定，可以先用 np.float32 试试；如果数值异常再换
DATA_TYPE = np.int16

# 3. 数据处理参数
SAMPLES_TO_DISCARD = 10000   # 丢弃的起始样本数（消除暂态）
SEGMENT_LENGTH = 1024        # 每个片段的样本长度
STRIDE = SEGMENT_LENGTH      # 步长，无重叠

# ===== 改动 3：输出文件夹名称改为 GMSK =====
CSV_OUTPUT_DIR = 'output_csv_files_1024_GMSK11'
PLOT_OUTPUT_DIR = 'output_plot_reports_1024_GMSK11'

# 要生成并保存分析图的样本数量
NUM_PLOTS_TO_SAVE = 2

# --- 创建输出文件夹 ---
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)


# --- 绘图函数（无需修改）---
def save_analysis_plot(iq_segment_transposed, segment_index, original_start_sample):
    """
    为给定的IQ片段生成并保存分析图。
    输入 iq_segment_transposed 的形状应为 (2, 1024)。
    """
    plt.ioff()
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    title = f'Analysis of Segment #{segment_index} | Original Start Sample: {original_start_sample}'
    fig.suptitle(title, fontsize=16)

    i_data = iq_segment_transposed[0, :]
    q_data = iq_segment_transposed[1, :]

    axs[0].plot(i_data, label='I component')
    axs[0].plot(q_data, label='Q component')
    axs[0].set_title('I/Q Time Domain Waveform')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Amplitude')

    axs[1].scatter(i_data, q_data, s=5)
    axs[1].set_title('Constellation Diagram')
    axs[1].grid(True)
    axs[1].set_aspect('equal', 'box')
    axs[1].set_xlabel('I (In-phase)')
    axs[1].set_ylabel('Q (Quadrature)')
    lim = np.max(np.abs(iq_segment_transposed)) * 1.2 + 1e-9
    axs[1].set_xlim(-lim, lim)
    axs[1].set_ylim(-lim, lim)

    complex_signal = i_data + 1j * q_data
    axs[2].psd(complex_signal, NFFT=SEGMENT_LENGTH, Fs=1)
    axs[2].set_title('Power Spectral Density (PSD)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f'segment_{segment_index:06d}_report.png'
    save_path = os.path.join(PLOT_OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)


# --- 主执行程序 ---
if __name__ == "__main__":

    # ===== 改动 4：检查两个文件是否存在 =====
    if not os.path.exists(I_FILE_PATH):
        print(f"Error: I-channel file not found at '{I_FILE_PATH}'")
        exit(1)
    if not os.path.exists(Q_FILE_PATH):
        print(f"Error: Q-channel file not found at '{Q_FILE_PATH}'")
        exit(1)

    try:
        # ===== 改动 5：分别加载 I 路和 Q 路数据 =====
        print(f"Step 1: Loading I/Q data from separate .dat files...")
        print(f"  I-channel: '{I_FILE_PATH}'")
        print(f"  Q-channel: '{Q_FILE_PATH}'")
        print(f"  Data type: {DATA_TYPE}")

        raw_i = np.fromfile(I_FILE_PATH, dtype=DATA_TYPE)
        raw_q = np.fromfile(Q_FILE_PATH, dtype=DATA_TYPE)

        print(f"  I-channel raw samples: {len(raw_i)}")
        print(f"  Q-channel raw samples: {len(raw_q)}")

        # ===== 改动 6：确保 I/Q 长度一致，取较短的长度 =====
        min_len = min(len(raw_i), len(raw_q))
        if len(raw_i) != len(raw_q):
            print(f"  Warning: I/Q lengths differ (I={len(raw_i)}, Q={len(raw_q)}). "
                  f"Truncating to {min_len} samples.")
        raw_i = raw_i[:min_len]
        raw_q = raw_q[:min_len]

        # ===== 改动 7：整数类型归一化 =====
        if np.issubdtype(DATA_TYPE, np.integer):
            max_val = np.iinfo(DATA_TYPE).max
            raw_i = raw_i.astype(np.float32) / max_val
            raw_q = raw_q.astype(np.float32) / max_val
            print(f"  Integer data normalized by max value: {max_val}")
        else:
            raw_i = raw_i.astype(np.float32)
            raw_q = raw_q.astype(np.float32)

        # ===== 改动 8：合并为 (N, 2) 的 IQ 对，与原流程对齐 =====
        iq_pairs = np.stack([raw_i, raw_q], axis=1)  # Shape: (N, 2)
        print(f"  Combined IQ pairs shape: {iq_pairs.shape}")

        # 清洗非有限值
        finite_mask = np.isfinite(iq_pairs).all(axis=1)
        clean_iq_data = iq_pairs[finite_mask]
        removed_count = iq_pairs.shape[0] - clean_iq_data.shape[0]
        if removed_count > 0:
            print(f"  Removed {removed_count} non-finite samples.")

        # 丢弃起始样本
        if clean_iq_data.shape[0] > SAMPLES_TO_DISCARD:
            stable_iq_data = clean_iq_data[SAMPLES_TO_DISCARD:]
        else:
            stable_iq_data = clean_iq_data
            print(f"  Warning: Data length ({clean_iq_data.shape[0]}) <= "
                  f"SAMPLES_TO_DISCARD ({SAMPLES_TO_DISCARD}). Using all data.")

        print(f"  Usable IQ samples after cleanup: {stable_iq_data.shape[0]}")

        # ===== 以下流程与原代码完全一致 =====

        # 2. 计算可切分的窗口总数
        num_segments = (stable_iq_data.shape[0] - SEGMENT_LENGTH) // STRIDE + 1
        if num_segments <= 0:
            raise ValueError(
                "Not enough data to form a single segment after cleaning and discarding. "
                f"Available: {stable_iq_data.shape[0]}, Required: {SEGMENT_LENGTH}"
            )

        print(f"\nStep 2: Will generate {num_segments} segments "
              f"(length={SEGMENT_LENGTH}, stride={STRIDE})")

        # ===== 新增：打印数据概览（调试用）=====
        print(f"\n  Data statistics:")
        print(f"    I-channel: min={stable_iq_data[:, 0].min():.6f}, "
              f"max={stable_iq_data[:, 0].max():.6f}, "
              f"mean={stable_iq_data[:, 0].mean():.6f}")
        print(f"    Q-channel: min={stable_iq_data[:, 1].min():.6f}, "
              f"max={stable_iq_data[:, 1].max():.6f}, "
              f"mean={stable_iq_data[:, 1].mean():.6f}")

        # 3. 遍历、保存 .csv 文件并生成样本图
        print(f"\nStep 3: Converting segments to .csv files...")

        for i in tqdm(range(num_segments), desc="Converting to .csv"):
            start_index = i * STRIDE
            end_index = start_index + SEGMENT_LENGTH

            segment_data = stable_iq_data[start_index:end_index]  # Shape: (1024, 2)

            # 转置为 (2, 1024) 格式：第一行 I，第二行 Q
            segment_to_save = segment_data.T  # Shape: (2, 1024)

            # 保存为 .csv 文件
            csv_filename = f'segment_{i:06d}.csv'
            csv_save_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)
            np.savetxt(csv_save_path, segment_to_save, delimiter=',')

            # 为前 N 个样本生成分析图
            if i < NUM_PLOTS_TO_SAVE:
                original_start_sample_pos = SAMPLES_TO_DISCARD + start_index
                save_analysis_plot(segment_to_save, i, original_start_sample_pos)

        print(f"\nConversion complete!")
        print(f"  {num_segments} .csv files saved in '{CSV_OUTPUT_DIR}'")
        print(f"  CSV shape: (2, {SEGMENT_LENGTH}) — Row 0: I, Row 1: Q")
        print(f"  {min(num_segments, NUM_PLOTS_TO_SAVE)} plot reports saved in '{PLOT_OUTPUT_DIR}'")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()