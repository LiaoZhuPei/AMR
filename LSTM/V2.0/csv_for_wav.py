import numpy as np
import os
from scipy.io.wavfile import write as write_wav
from tqdm import tqdm

# --- 配置区 ---

# 1. 输入的大型CSV文件名 (由V18 MATLAB脚本生成)
INPUT_CSV_FILE = 'cpc2410_APSK_16_100MB.csv'

# 2. 输出文件名 (不带扩展名)
OUTPUT_FILENAME_BASE = 'cpc2410_APSK_16_100MB'

# 3. WAV文件参数
#    采样率 (Hz) - 这是一个元数据，您可以根据需要设置。
#    48000 Hz 是一个标准的音频采样率。
WAV_SAMPLE_RATE = 48000

# --- 主执行程序 ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"错误: 找不到输入的CSV文件 '{INPUT_CSV_FILE}'")
    else:
        try:
            # --- 步骤 1: 加载并重构CSV数据 ---
            print(f"步骤 1: 正在从 '{INPUT_CSV_FILE}' 加载并重构数据...")
            
            # 逐行读取文件以处理MATLAB fprintf导致的多行格式
            with open(INPUT_CSV_FILE, 'r') as f:
                lines = f.readlines()

            i_lines = []
            q_lines = []
            # 将奇数行（I）和偶数行（Q）分开
            for i, line in tqdm(enumerate(lines), total=len(lines), desc="正在解析行"):
                cleaned_line = line.strip().strip(',')
                if cleaned_line: # 确保不是空行
                    if i % 2 == 0: # 奇数行 (0, 2, 4...) 是 I
                        i_lines.append(cleaned_line)
                    else: # 偶数行 (1, 3, 5...) 是 Q
                        q_lines.append(cleaned_line)
            
            # 拼接并从字符串解析出numpy数组
            i_data = np.fromstring(','.join(i_lines), sep=',')
            q_data = np.fromstring(','.join(q_lines), sep=',')
            
            # 将I和Q数据堆叠成一个 (2, N) 的数组，数据类型为 float
            iq_data_float = np.vstack([i_data, q_data])
            print(f"数据加载和重构成功！最终数组形状为: {iq_data_float.shape}")

            # --- 步骤 2: 将浮点数据缩放并转换为 int16 ---
            print("\n步骤 2: 正在将数据缩放至 16位整数 (int16) 范围...")
            # 首先确保数据在 [-1.0, 1.0] 范围内
            max_abs = np.max(np.abs(iq_data_float))
            if max_abs > 1.0:
                iq_data_float = iq_data_float / max_abs
            
            # 缩放并转换类型
            iq_data_int16 = (iq_data_float * 32767).astype(np.int16)
            print("数据转换成功！")

            # --- 步骤 3: 保存为 .wav 文件 ---
            wav_filename = f"{OUTPUT_FILENAME_BASE}.wav"
            print(f"\n步骤 3: 正在保存为 .wav 文件: '{wav_filename}'...")
            
            # scipy.io.wavfile.write 需要 (N, 2) 的形状，代表 N个采样点，2个通道
            # 我们当前的形状是 (2, N)，所以需要进行转置 (.T)
            wav_data = iq_data_int16.T
            
            write_wav(wav_filename, WAV_SAMPLE_RATE, wav_data)
            print(f".wav 文件保存成功！(采样率: {WAV_SAMPLE_RATE} Hz, 格式: 16-bit 立体声)")

            # --- 步骤 4: 保存为 .iq 文件 ---
            iq_filename = f"{OUTPUT_FILENAME_BASE}.iq"
            print(f"\n步骤 4: 正在保存为 .iq 文件: '{iq_filename}'...")
            
            # .iq 文件需要 I, Q 交错。
            # wav_data 的形状 (N, 2) 已经是交错的正确内存布局
            # 直接将其写入文件即可
            wav_data.tofile(iq_filename)
            print(f".iq 文件保存成功！(格式: 16-bit 整数, I/Q 交错)")

            print("\n所有转换任务完成！")

        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")