import h5py
import numpy as np
import os

mat_folder = 'E:/Project/AMR-Benchmark/output_mat_files_GMSK5/'
csv_folder = 'E:/C_CNN_ONNX/onnxruntime_AMR_x/data_GMSK5/'

mat_files = sorted([f for f in os.listdir(mat_folder) if f.endswith('.mat')])
csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith('.csv')])

print(f"mat 文件数: {len(mat_files)}")
print(f"csv 文件数: {len(csv_files)}")
print(f"\nmat 前5个文件名: {mat_files[:5]}")
print(f"csv 前5个文件名: {csv_files[:5]}")
print(f"\nmat 后5个文件名: {mat_files[-5:]}")
print(f"csv 后5个文件名: {csv_files[-5:]}")