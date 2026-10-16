import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
# --- 导入正确的 LSTM 模型结构 ---
import LSTM as lstm_model

# --- 1. 基本配置 ---
classes = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

# --- 修改点 1: 更新模型路径 ---
# 请确保这里的文件名与你训练时保存的文件名一致
MODEL_WEIGHTS_PATH = 'weights/pytorch_lstm_iq_rms_norm.pt' 

# 指定包含测试文件的文件夹路径
TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_part_2FSK/'
# 指定要测试的样本数量 (N)
NUM_SAMPLES_TO_TEST = 100
# 指定文件扩展名
FILE_EXTENSION = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. 核心函数 ---

# --- 修改点 2: 使用 RMS (功率) 归一化函数 ---
def per_sample_normalize_single(signal_data):
    """
    对单个样本进行功率归一化 (RMS Normalization)。
    使得信号的复数平均功率 E[|s|^2] = 1。
    
    输入 shape: (1, 2, 500) -> 对应 (Batch=1, Channels=2, Length=500)
    """
    # 1. 确保数据是浮点型
    signal_data = signal_data.astype(np.float32)
    
    # 2. 计算瞬时功率 P = I^2 + Q^2
    # signal_data[:, 0, :] 是 I路, signal_data[:, 1, :] 是 Q路
    # 结果 shape 为 (1, 500)
    instantaneous_power = signal_data[:, 0, :]**2 + signal_data[:, 1, :]**2
    
    # 3. 计算平均功率 (Scalar)
    avg_power = np.mean(instantaneous_power)
    
    # 4. 防止除以 0
    if avg_power < 1e-10:
        return np.zeros_like(signal_data)
    
    # 5. 计算 RMS 值
    rms_scale = np.sqrt(avg_power)
    
    # 6. 归一化
    normalized_data = signal_data / rms_scale
    
    return normalized_data


def predict_single_signal(model, signal_data, device, class_list):
    """
    对单个经过归一化的 I/Q 信号 (NumPy array) 进行预测。
    输入信号数据形状应为 (1, 2, 500)。
    """
    if signal_data.shape != (1, 2, 500):
        raise ValueError(f"Expected signal_data shape (1, 2, 500), but got {signal_data.shape}")

    model.eval()

    # 1. 将 NumPy 数组转换为 PyTorch Tensor
    input_tensor = torch.from_numpy(signal_data).float()

    # 2. 调整维度以匹配 LSTM 输入 (N, C, L) -> (N, L, C)
    #    (1, 2, 500) -> (1, 500, 2)
    input_tensor = input_tensor.permute(0, 2, 1)

    # 3. 将 Tensor 移动到正确的设备
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)
    confidence, predicted_idx_tensor = torch.max(probabilities, 1)

    predicted_idx = predicted_idx_tensor.item()
    confidence_score = confidence.item()
    predicted_class = class_list[predicted_idx]

    return predicted_class, confidence_score


# --- 3. 主执行程序 ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at '{MODEL_WEIGHTS_PATH}'")
        print("Please check if the filename matches what you saved during training.")
    else:
        # a. 加载 LSTM 模型
        print("Loading LSTM model architecture...")
        model = lstm_model.build_model(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        print("Model loaded successfully.")

        # b. 查找并准备测试文件
        print(f"\nScanning for '{FILE_EXTENSION}' files in '{TEST_DATA_FOLDER}'...")
        search_pattern = os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}')
        all_files = glob.glob(search_pattern)

        if not all_files:
            print(f"Error: No '{FILE_EXTENSION}' files found in the specified folder.")
            exit()

        files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
        print(f"Found {len(all_files)} files. Will test the first {len(files_to_test)} samples.")

        # c. 循环预测并统计结果
        prediction_counts = Counter()
        print("\n" + "=" * 50)
        print("STARTING BATCH PREDICTION (LSTM with RMS Normalization)")
        print("=" * 50)

        for i, file_path in enumerate(files_to_test):
            print(f"\n--- Processing Sample {i + 1}/{len(files_to_test)}: {os.path.basename(file_path)} ---")

            try:
                # 1. 加载单个信号文件
                with h5py.File(file_path, 'r') as data_file:
                    single_signal_sample = data_file['data_save'][:]

                # 2. 检查数据形状
                if single_signal_sample.shape != (1, 2, 500):
                    print(f"  [Skipping] Incorrect data shape: {single_signal_sample.shape}. Expected (1, 2, 500).")
                    continue

                # --- 修改点 3: 应用 RMS 归一化预处理 ---
                normalized_signal_sample = per_sample_normalize_single(single_signal_sample)

                # 4. 调用预测函数
                predicted_modulation, confidence = predict_single_signal(
                    model=model,
                    signal_data=normalized_signal_sample,  # 传递处理后的 NumPy 数组
                    device=device,
                    class_list=classes
                )

                # 5. 打印结果和更新计数
                print(f"  -> Predicted Modulation: {predicted_modulation}")
                print(f"  -> Confidence: {confidence * 100:.2f}%")
                prediction_counts[predicted_modulation] += 1

            except FileNotFoundError:
                print(f"  [Error] Test data file not found at '{file_path}'")
            except Exception as e:
                print(f"  [Error] An error occurred while processing {os.path.basename(file_path)}: {e}")

        # d. 打印最终的统计摘要
        print("\n" + "=" * 50)
        print("                PREDICTION SUMMARY")
        print("=" * 50)

        total_processed = sum(prediction_counts.values())
        print(f"Total samples successfully processed: {total_processed}/{len(files_to_test)}")

        if not prediction_counts:
            print("No successful predictions were made.")
        else:
            print("\nPrediction Counts:")
            for class_name in classes:
                count = prediction_counts.get(class_name, 0)
                print(f"- {class_name:<10}: {count} times")

        print("=" * 50)