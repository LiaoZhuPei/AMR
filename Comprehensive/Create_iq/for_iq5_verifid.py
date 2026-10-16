import h5py
import os

# 要检查的文件夹和文件名
MAT_OUTPUT_DIR = 'output_mat_files'
FILENAME_TO_CHECK = 'segment_000000.mat'
DATASET_NAME = 'data_save'

file_path = os.path.join(MAT_OUTPUT_DIR, FILENAME_TO_CHECK)

if os.path.exists(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f[DATASET_NAME]
        print(f"Successfully loaded '{file_path}'.")
        print(f"The shape of the dataset '{DATASET_NAME}' is: {data.shape}")

        if data.shape == (2, 1024):
            print("Verification successful! The shape is correct.")
        else:
            print(f"Verification FAILED! Expected shape (1, 2, 1024), but got {data.shape}.")
else:
    print(f"Error: File '{file_path}' not found.")