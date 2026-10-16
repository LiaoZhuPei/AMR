import h5py
import numpy as np

mat_path = 'E:/Project/AMR-Benchmark/output_mat_files_GMSK5/segment_000099.mat'
csv_path = 'E:/C_CNN_ONNX/onnxruntime_AMR_x/data_GMSK5/segment_000099.csv'  # 对应的csv路径

# 从 mat 读
with h5py.File(mat_path, 'r') as f:
    data = f['data_save'][:]  # shape (1, 2, 1024)
i_mat = data[0, 0, :]  # float32
q_mat = data[0, 1, :]  # float32

# 从 csv 读
with open(csv_path, 'r') as f:
    lines = f.readlines()
i_csv = np.array([float(x) for x in lines[0].strip().split(',')], dtype=np.float32)
q_csv = np.array([float(x) for x in lines[1].strip().split(',')], dtype=np.float32)

# 对比
print(f"I_mat[0:5] = {i_mat[:5]}")
print(f"I_csv[0:5] = {i_csv[:5]}")
print(f"Q_mat[0:5] = {q_mat[:5]}")
print(f"Q_csv[0:5] = {q_csv[:5]}")
print(f"I 最大差异: {np.max(np.abs(i_mat - i_csv)):.10f}")
print(f"Q 最大差异: {np.max(np.abs(q_mat - q_csv)):.10f}")
print(f"完全一致: {np.array_equal(i_mat, i_csv) and np.array_equal(q_mat, q_csv)}")