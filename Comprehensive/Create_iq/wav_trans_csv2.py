import numpy as np
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt


def create_dummy_iq_wav(filename="dummy_iq_signal.wav", duration_sec=5, sample_rate=44100):
    """
    创建一个虚拟的IQ信号 .wav 文件用于测试。
    信号是一个简单的 QPSK 调制信号加上一些噪声。
    """
    if os.path.exists(filename):
        print(f"'{filename}' 文件已存在，跳过创建。")
        return

    print(f"正在创建虚拟 IQ .wav 文件: '{filename}'...")
    num_samples = int(duration_sec * sample_rate)

    # 生成一些随机的 QPSK 符号
    num_symbols = int(np.ceil(num_samples / 100))
    # 使用较大的幅度以模拟原始 int16 值
    symbols_i = np.random.choice([-20000, 20000], num_symbols)
    symbols_q = np.random.choice([-20000, 20000], num_symbols)

    # 通过重复符号来生成波形
    i_data = np.repeat(symbols_i, 100)[:num_samples]
    q_data = np.repeat(symbols_q, 100)[:num_samples]

    # 添加一些高斯白噪声
    noise = np.random.normal(0, 500, num_samples)
    i_data = i_data + noise
    q_data = q_data + noise

    # 组合成 (N, 2) 的形式
    iq_interleaved = np.stack([i_data, q_data], axis=1)

    # 确保值在 int16 范围内
    iq_clipped = np.clip(iq_interleaved, -32767, 32767)

    # 转换为 int16
    iq_int16 = iq_clipped.astype(np.int16)

    # 写入 .wav 文件
    wavfile.write(filename, sample_rate, iq_int16)
    print(f"'{filename}' 创建成功。")


def process_iq_wav_raw_amplitude(wav_file_path, output_dir="output", num_segments=200, segment_length=1024):
    """
    将 IQ .wav 文件拆分为多个 .csv 片段和波形图，保留原始幅度值。

    参数:
    - wav_file_path (str): 输入的 .wav 文件路径。
    - output_dir (str): 输出文件存放的根目录。
    - num_segments (int): 要拆分的片段数量。
    - segment_length (int): 每个片段的长度。
    """
    # --- 1. 准备输出目录 ---
    csv_dir = os.path.join(output_dir, "csv_files")
    plot_dir = os.path.join(output_dir, "plot_images")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"输出将保存到 '{output_dir}' 目录中。")

    # --- 2. 读取并处理 WAV 文件 ---
    try:
        sample_rate, data = wavfile.read(wav_file_path)
        print(f"成功读取 '{wav_file_path}', 采样率: {sample_rate}, 数据形状: {data.shape}, 数据类型: {data.dtype}")
    except FileNotFoundError:
        print(f"错误: 文件 '{wav_file_path}' 未找到。")
        return

    # 检查数据是否为立体声（IQ两路）
    if data.ndim != 2 or data.shape[1] != 2:
        print(f"错误: WAV 文件必须是立体声（包含两路数据）。当前数据形状为 {data.shape}")
        return

    # 检查数据总长度是否足够
    total_required_samples = num_segments * segment_length
    if len(data) < total_required_samples:
        print(f"错误: 数据长度不足。需要 {total_required_samples} 个采样点, 但文件只有 {len(data)} 个。")
        return

    # --- 3. 数据类型转换 (移除归一化) ---
    data_float = data.astype(np.float32)
    print("数据已转换为浮点类型，保留原始幅度。")

    # wav 文件通常是 (N, 2) 格式, N是采样点数
    # 我们需要将其转换为 (2, N) 格式，第一行是I，第二行是Q
    iq_data = data_float.T
    print(f"数据已转置为 (I, Q) 格式，新形状: {iq_data.shape}")

    # --- 4. 循环切片、保存和绘图 ---
    for i in range(num_segments):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = iq_data[:, start_index:end_index]

        if segment.shape != (2, segment_length):
            print(f"警告: 第 {i + 1} 个片段形状异常 {segment.shape}，跳过。")
            continue

        base_filename = f"segment_{i + 1:03d}"
        csv_path = os.path.join(csv_dir, f"{base_filename}.csv")
        plot_path = os.path.join(plot_dir, f"{base_filename}.png")

        np.savetxt(csv_path, segment, delimiter=',')

        i_channel = segment[0, :]
        q_channel = segment[1, :]
        time_axis = np.arange(segment_length)

        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, i_channel, label='I-Channel', color='blue')
        plt.plot(time_axis, q_channel, label='Q-Channel', color='red', alpha=0.8)
        plt.title(f"IQ Waveform - {base_filename}")
        plt.xlabel("Sample Index")
        plt.ylabel("Original Amplitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()

        if (i + 1) % 20 == 0 or (i + 1) == num_segments:
            print(f"已处理 {i + 1}/{num_segments} 个片段...")

    print("\n处理完成！")
    print(f"CSV 文件保存在: {os.path.abspath(csv_dir)}")
    print(f"波形图保存在: {os.path.abspath(plot_dir)}")


if __name__ == "__main__":
    # --- 用户配置 ---
    create_dummy_iq_wav()

    WAV_FILE = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'

    OUTPUT_DIRECTORY = 'wav_iq_data_output_raw_amplitude'
    NUM_OF_SEGMENTS = 200
    LENGTH_OF_SEGMENT = 1024

    # --- 运行主函数 ---
    process_iq_wav_raw_amplitude(
        wav_file_path=WAV_FILE,
        output_dir=OUTPUT_DIRECTORY,
        num_segments=NUM_OF_SEGMENTS,
        # 【修正点】: 将参数名从 length_of_segment 改为 segment_length
        segment_length=LENGTH_OF_SEGMENT
    )