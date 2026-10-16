import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
# 导入您自定义的CNN模型结构
import CNNLSTM as mcl


# --- 1. 基本配置 ---
classes = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']
# PI4DQPSK
# 模型权重路径保持不变
MODEL_WEIGHTS_PATH = 'weights/weights_cnnlstm_mata1_3.pt'

# 指定包含测试文件的文件夹路径
TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_QAM64/'
# 指定要测试的样本数量 (N)
NUM_SAMPLES_TO_TEST = 100
# 指定文件扩展名，以防文件夹中有其他类型的文件
FILE_EXTENSION = '.mat'  # 如果是csv，则改为 '.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def per_sample_normalize_single(signal_data):

    #对单个样本（独立的时间序列）进行归一化，将其缩放到 [-1, 1] 区间。

    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data)
    #normalized_data = (signal_data - min_val) / range_val
    normalized_data = 2*(signal_data - min_val) / range_val - 1
    return normalized_data

def predict_single_signal(model, signal_data, device, class_list):
    """
    对单个信号进行调制识别预测。
    输入信号数据形状应为 (1, 2, 1024)。
    """
    if signal_data.shape != (1, 2, 1024):
        raise ValueError(f"Expected signal_data shape (1, 2, 1024), but got {signal_data.shape}")

    model.eval()
    # 预处理: NumPy (1, 2, 1024) -> Tensor (1, 1, 2, 1024) [增加通道维度]
    input_tensor = torch.from_numpy(signal_data).float()
    input_tensor = input_tensor.unsqueeze(0).to(device)

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
    else:
        # a. 加载模型 (只需加载一次)
        print("Loading model architecture...")
        model = mcl.build_model(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        print("Model loaded successfully.")

        # --- 修改点 2: 查找并准备测试文件 ---
        print(f"\nScanning for '{FILE_EXTENSION}' files in '{TEST_DATA_FOLDER}'...")

        # 使用 glob 查找所有匹配的文件
        search_pattern = os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}')
        all_files = glob.glob(search_pattern)

        if not all_files:
            print(f"Error: No '{FILE_EXTENSION}' files found in the specified folder.")
            exit()

        # 根据 NUM_SAMPLES_TO_TEST 截取文件列表
        files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
        print(f"Found {len(all_files)} files. Will test the first {len(files_to_test)} samples.")

        # --- 修改点 3: 循环预测并统计结果 ---
        prediction_counts = Counter()  # 用于统计各类别的预测次数

        print("\n" + "=" * 50)
        print("STARTING BATCH PREDICTION")
        print("=" * 50)

        for i, file_path in enumerate(files_to_test):
            print(f"\n--- Processing Sample {i + 1}/{len(files_to_test)}: {os.path.basename(file_path)} ---")

            try:
                # b. 加载单个信号文件
                # 适用于 .mat 文件
                with h5py.File(file_path, 'r') as data_file:
                    single_signal_sample = data_file['data_save'][:]

                # 适用于 .csv 文件 (如果需要，取消注释此部分)
                # df = pd.read_csv(file_path, header=None)
                # single_signal_sample = df.to_numpy()
                # # 如果csv文件是(2,1024),需要reshape
                # if single_signal_sample.shape == (2, 1024):
                #     single_signal_sample = single_signal_sample.reshape(1, 2, 1024)

                # 确保数据形状是 (1, 2, 1024)
                if single_signal_sample.shape != (1, 2, 1024):
                    print(f"  [Skipping] Incorrect data shape: {single_signal_sample.shape}. Expected (1, 2, 1024).")
                    continue  # 跳过此文件，继续下一个

                # c. 应用逐样本归一化
                normalized_signal_sample = per_sample_normalize_single(single_signal_sample)

                # d. 调用预测函数
                predicted_modulation, confidence = predict_single_signal(
                    model=model,
                    signal_data=normalized_signal_sample,
                    device=device,
                    class_list=classes
                )

                # e. 打印单个结果并更新计数器
                print(f"  -> Predicted Modulation: {predicted_modulation}")
                print(f"  -> Confidence: {confidence * 100:.2f}%")
                prediction_counts[predicted_modulation] += 1

            except FileNotFoundError:
                print(f"  [Error] Test data file not found at '{file_path}'")
            except Exception as e:
                print(f"  [Error] An error occurred while processing {os.path.basename(file_path)}: {e}")

        # --- 修改点 4: 打印最终的统计摘要 ---
        print("\n" + "=" * 50)
        print("                PREDICTION SUMMARY")
        print("=" * 50)

        total_processed = sum(prediction_counts.values())
        print(f"Total samples successfully processed: {total_processed}/{len(files_to_test)}")

        if not prediction_counts:
            print("No successful predictions were made.")
        else:
            print("\nPrediction Counts:")
            # 打印所有可能类别的计数，即使是0
            for class_name in classes:
                count = prediction_counts.get(class_name, 0)
                print(f"- {class_name:<10}: {count} times")

        print("=" * 50)