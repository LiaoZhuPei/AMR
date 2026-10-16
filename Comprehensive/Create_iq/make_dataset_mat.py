import h5py
import numpy as np
import glob
import os
import scipy.io as sio
import csv

# --- 1. 设置路径和文件模式 ---
# !! 修改这里为你存放 .mat 文件的文件夹路径 !!
data_folder = 'E:/M_Project/mat/test'  # 示例路径
file_pattern = os.path.join(data_folder, 'test_*.mat')

file_paths = sorted(glob.glob(file_pattern))
if not file_paths:
    raise FileNotFoundError(f"在文件夹 '{data_folder}' 中没有找到任何匹配 '{file_pattern}' 的文件。")

print(f"找到 {len(file_paths)} 个数据文件，将按以下顺序合并：")
for path in file_paths:
    print(f"  - {os.path.basename(path)}")

# --- 2. 加载并拼接所有数据块 ---
all_data_list = []
samples_per_class = []

for path in file_paths:
    print(f"正在加载: {os.path.basename(path)}...")
    try:
        with h5py.File(path, 'r') as f:
            # 读取原始数据，此时形状是 (L, 2, N)
            data_chunk_raw = f['data_save'][:]

            # !! 关键修正：进行维度转置 !!
            # 从 (L, 2, N) 转置为 (N, 2, L)
            # 例如从 (1024, 2, 500) 转置为 (500, 2, 1024)
            data_chunk_transposed = np.transpose(data_chunk_raw, (2, 1, 0))

            # 现在使用转置后的数据进行后续操作
            all_data_list.append(data_chunk_transposed)

            # 获取正确的样本数 (N)，它现在是第 0 维
            num_samples = data_chunk_transposed.shape[0]
            samples_per_class.append(num_samples)

            print(f"  > 原始形状: {data_chunk_raw.shape}, 转置后形状: {data_chunk_transposed.shape}")
            print(f"  > 成功加载了 {num_samples} 个样本。")

    except OSError as e:
        print(f"  > 错误! 无法打开文件 '{os.path.basename(path)}'. 错误信息: {e}")
        raise
    except KeyError:
        print(f"  > 错误! 文件 '{os.path.basename(path)}' 中没有找到名为 'data_save' 的变量。")
        raise

# --- 3. 拼接、创建标签和保存 (这部分逻辑无需修改) ---

print("\n正在内存中拼接所有数据...")
# 沿着第0轴（样本轴）拼接，现在是正确的
combined_data = np.concatenate(all_data_list, axis=0)
print(f"数据拼接完成！最终数据形状: {combined_data.shape}")

print("\n正在创建标签...")
labels_list = []
for class_index, num_samples in enumerate(samples_per_class):
    labels_list.extend([class_index] * num_samples)
final_labels = np.array(labels_list).reshape(-1, 1)
print(f"标签创建完成！最终标签形状: {final_labels.shape}")

if combined_data.shape[0] == final_labels.shape[0]:
    print("\n成功！数据样本数和标签数匹配！")
else:
    raise ValueError("严重错误：数据样本数与标签数不匹配！")

output_folder = data_folder
output_mat_path = os.path.join(output_folder, 'test.mat')
print(f"\n正在将数据保存到 .mat 文件: {output_mat_path}")
sio.savemat(
    output_mat_path,
    {'data_save': combined_data},
    do_compression=True,
    long_field_names=True
)
print("  > .mat 文件保存成功！")

output_csv_path = os.path.join(output_folder, 'test_labels.csv')
print(f"正在将标签保存到 .csv 文件: {output_csv_path}")
np.savetxt(output_csv_path, final_labels, fmt='%d', delimiter=',')
print("  > .csv 文件保存成功！")

print("\n所有操作完成！")
