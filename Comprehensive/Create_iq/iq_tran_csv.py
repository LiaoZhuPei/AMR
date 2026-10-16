import numpy as np
import os
import matplotlib.pyplot as plt


def create_dummy_iq_file(filename="dummy_signal.iq", num_samples=500000, dtype=np.float32):
    """
    创建一个虚拟的 .iq 文件用于测试。
    这是一个二进制文件，包含交错的 I 和 Q 数据。
    """
    if os.path.exists(filename):
        print(f"'{filename}' 文件已存在，跳过创建。")
        return

    print(f"正在创建虚拟 IQ .iq 文件: '{filename}'...")
    # 生成一些简单的 BPSK 信号
    bits = np.random.randint(0, 2, num_samples // 100) * 2 - 1
    i_data = np.repeat(bits, 100).astype(np.float32)
    q_data = np.zeros(len(i_data), dtype=np.float32)  # BPSK 的 Q 通道为 0

    # 添加噪声
    i_data += np.random.normal(0, 0.1, len(i_data))
    q_data += np.random.normal(0, 0.1, len(q_data))

    # 交错 I 和 Q 数据
    iq_interleaved = np.empty(i_data.size + q_data.size, dtype=dtype)
    iq_interleaved[0::2] = i_data
    iq_interleaved[1::2] = q_data

    # 写入二进制文件
    iq_interleaved.tofile(filename)
    print(f"'{filename}' 创建成功。")


def read_iq_file(file_path, data_type=np.complex64):
    """
    从 .iq 文件中读取原始 IQ 数据。

    参数:
    - file_path (str): .iq 文件路径。
    - data_type: NumPy 数据类型 (例如 np.complex64, np.float32, np.int16)。
                 如果数据是 I,Q 交错的非复数类型，处理方式会不同。

    返回:
    - (N, 2) 形状的 NumPy 数组，N 是 IQ 采样点数。
    """
    try:
        data = np.fromfile(file_path, dtype=data_type)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None

    # 如果是复数类型，直接处理
    if np.iscomplexobj(data):
        # 将 (N,) 的复数数组转换为 (N, 2) 的实数数组
        iq_data = np.vstack((data.real, data.imag)).T
        return iq_data

    # 如果是 I, Q 交错的实数类型 (float32, int16, etc.)
    else:
        # 检查样本数量是否为偶数
        if data.size % 2 != 0:
            print(f"警告: 文件 '{file_path}' 的样本总数是奇数，最后一个样本将被丢弃。")
            data = data[:-1]
        # 将 [I1, Q1, I2, Q2, ...] 重塑为 [[I1, Q1], [I2, Q2], ...]
        iq_data = data.reshape(-1, 2)
        return iq_data


def process_iq_data(input_file_path, data_type, output_dir="output", num_segments=200, segment_length=1024):
    """
    将 IQ 数据文件拆分为多个 .csv 片段和波形图。

    参数:
    - input_file_path (str): 输入的 .iq 文件路径。
    - data_type: IQ 文件的数据类型。
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

    # --- 2. 读取 IQ 文件 ---
    data = read_iq_file(input_file_path, data_type=data_type)
    if data is None:
        return
    print(f"成功读取 '{input_file_path}', 数据形状: {data.shape}")

    # --- 3. 数据归一化和格式转换 ---
    # 检查数据总长度是否足够
    total_required_samples = num_segments * segment_length
    if len(data) < total_required_samples:
        print(f"错误: 数据长度不足。需要 {total_required_samples} 个采样点, 但文件只有 {len(data)} 个。")
        return

    # 进行全局幅度归一化
    # 如果是整数类型，基于类型最大值归一化
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data_float = data.astype(np.float32) / max_val
    # 如果是浮点数类型，基于数据中的最大绝对值归一化
    else:
        max_abs_val = np.max(np.abs(data))
        if max_abs_val == 0:
            data_float = data.astype(np.float32)  # 避免除以零
        else:
            data_float = data.astype(np.float32) / max_abs_val

    # 转置为 (2, N) 格式
    iq_data = data_float.T
    print(f"数据已归一化并转置为 (I, Q) 格式，新形状: {iq_data.shape}")

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

        plt.figure(figsize=(10, 6))
        plt.plot(segment[0, :], label='I-Channel', color='blue')
        plt.plot(segment[1, :], label='Q-Channel', color='red', alpha=0.8)
        plt.title(f"IQ Waveform - {base_filename}")
        plt.xlabel("Sample Index")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.grid(True)
        plt.ylim(-1.1, 1.1)
        plt.savefig(plot_path)
        plt.close()

        if (i + 1) % 20 == 0 or (i + 1) == num_segments:
            print(f"已处理 {i + 1}/{num_segments} 个片段...")

    print("\n处理完成！")
    print(f"CSV 文件保存在: {os.path.abspath(csv_dir)}")
    print(f"波形图保存在: {os.path.abspath(plot_dir)}")


if __name__ == "__main__":
    # --- 用户配置 ---
    # 1. 指定你的 .iq 文件的数据类型
    #    常见的选项:
    #    - np.complex64: 复数 (I+jQ)，每个样本 8 字节。GNU Radio 常用格式。
    #    - np.float32:   I, Q 交错的32位浮点数。
    #    - np.int16:     I, Q 交错的16位整数。
    #    - np.int8:      I, Q 交错的8位整数。
    #    !!!!!! 你必须知道你的 .iq 文件是用哪种格式保存的 !!!!!!
    DATA_TYPE = np.int16

    # 2. 如果没有 .iq 文件，脚本会创建一个用于测试。
    create_dummy_iq_file("QAM16-03-24-25-10h52m09s762.iq", dtype=DATA_TYPE)

    # 3. 将此路径替换为您自己的 .iq 文件路径
    IQ_FILE = 'QAM16-03-24-25-10h52m09s762.iq'  # <--- 修改这里

    # 4. 设置输出目录和参数
    OUTPUT_DIRECTORY = 'QAM16_500_CSV'
    NUM_OF_SEGMENTS = 200
    LENGTH_OF_SEGMENT = 500

    # --- 运行主函数 ---
    process_iq_data(
        input_file_path=IQ_FILE,
        data_type=DATA_TYPE,
        output_dir=OUTPUT_DIRECTORY,
        num_segments=NUM_OF_SEGMENTS,
        segment_length=LENGTH_OF_SEGMENT
    )