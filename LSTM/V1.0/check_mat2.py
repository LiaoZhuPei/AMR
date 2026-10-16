import h5py

# --- 修改这里，指向你的 .mat 文件 ---
mat_file_path = 'E:/Project/AMR-Benchmark/HisarMod/dataset/HisarModdataset/Test/test.mat'

try:
    with h5py.File(mat_file_path, 'r') as f:
        print(f"Keys and shapes inside '{mat_file_path}':")
        for key in f.keys():
            shape = f[key].shape
            print(f"- Key: '{key}', Shape: {shape}")
            # 帮助你分析的提示
            if 260000 in shape:
                print(f"  ^^^ This key likely contains your data. Note the order of dimensions.")

except Exception as e:
    print(f"An error occurred: {e}")