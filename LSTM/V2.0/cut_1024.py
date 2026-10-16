import numpy as np
import os
from tqdm import tqdm

# --- 配置区 ---
INPUT_CSV_FILE = 'APSK_16_100MB_output.csv'
OUTPUT_DIR = 'output_segments_csv_16'
NUM_FILES_TO_CREATE = 100
SEGMENT_LENGTH = 1024
# --- ---

os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"错误: 找不到输入的CSV文件 '{INPUT_CSV_FILE}'")
    else:
        try:
            print(f"步骤 1: 正在从 '{INPUT_CSV_FILE}' 加载数据...")
            # 由于MATLAB的fprintf每次追加都会换行，我们需要特殊处理
            # 我们逐行读取文件
            with open(INPUT_CSV_FILE, 'r') as f:
                lines = f.readlines()

            i_lines = []
            q_lines = []
            # 将奇数行（I）和偶数行（Q）分开
            for i, line in enumerate(lines):
                # 去除行尾多余的逗号和换行符
                cleaned_line = line.strip().strip(',')
                if i % 2 == 0: # 奇数行 (0, 2, 4...)
                    i_lines.append(cleaned_line)
                else: # 偶数行 (1, 3, 5...)
                    q_lines.append(cleaned_line)
            
            # 将所有I行和Q行的数据拼接成一个长字符串
            i_full_string = ','.join(i_lines)
            q_full_string = ','.join(q_lines)
            
            # 从字符串中解析出numpy数组
            i_data = np.fromstring(i_full_string, sep=',')
            q_data = np.fromstring(q_full_string, sep=',')
            
            # 将I和Q数据堆叠成一个 (2, N) 的数组
            all_data = np.vstack([i_data, q_data])
            
            print(f"数据加载和重构成功！最终数组形状为: {all_data.shape}")

            if all_data.ndim < 2 or all_data.shape[0] != 2:
                raise ValueError("重构后的数据不是预期的2行格式！")

            max_possible_files = all_data.shape[1] // SEGMENT_LENGTH
            num_to_create = min(NUM_FILES_TO_CREATE, max_possible_files)

            if num_to_create < NUM_FILES_TO_CREATE:
                print(f"警告: 数据不足以创建 {NUM_FILES_TO_CREATE} 个文件，将只创建 {num_to_create} 个。")
            
            print(f"\n步骤 2: 开始截取并保存 {num_to_create} 个片段...")
            
            for i in tqdm(range(num_to_create), desc="正在创建文件"):
                start_col = i * SEGMENT_LENGTH
                end_col = start_col + SEGMENT_LENGTH
                
                segment = all_data[:, start_col:end_col]
                
                output_filename = os.path.join(OUTPUT_DIR, f'segment_{i:03d}.csv')
                
                np.savetxt(output_filename, segment, delimiter=',', fmt='%.8f')

            print(f"\n处理完成！")
            print(f"总共在 '{OUTPUT_DIR}' 文件夹中创建了 {num_to_create} 个文件。")

        except Exception as e:
            print(f"\n处理过程中发生错误: {e}")