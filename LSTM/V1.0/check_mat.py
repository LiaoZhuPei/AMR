import h5py

# --- 修改这里，指向你的 .mat 文件 ---
mat_file_path = 'E:/Project/AMR-Benchmark/HisarMod/dataset/HisarModdataset/Test/test.mat'

try:
    with h5py.File(mat_file_path, 'r') as f:
        print(f"Keys inside '{mat_file_path}':")
        # 打印文件顶层的所有键
        keys = list(f.keys())
        print(keys)

        # 打印每个键对应的数据形状，帮助你识别
        for key in keys:
            print(f"- Key: '{key}', Shape: {f[key].shape}")

except Exception as e:
    print(f"An error occurred: {e}")