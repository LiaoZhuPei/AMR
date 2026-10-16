import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py

# ===== 改动 1：导入新模型 =====
from test01_model import build_model  # 替换原来的 import CNNLSTM as mcl

# --- 1. 基本配置 ---
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM','2ASK','Pi4DQPSK', '64QAM', '32QAM','128QAM','256QAM', 'MSK']
# PI4DQPSK
# ===== 改动 2：权重路径对应 train.py 中保存的文件名 =====
MODEL_WEIGHTS_PATH = 'weights/cnn_bilstm_attn_matd1_1.pt'

TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_BPSK/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===== 改动 3：使用功率归一化，与 train.py 中的预处理完全一致 =====
def power_normalize(signal_data):
    """
    功率归一化：将信号平均功率归一化为 1
    输入 shape: (2, 1024) 或 (1, 2, 1024)
    必须与训练时的预处理完全一致！
    """
    I = signal_data[..., 0, :] if signal_data.ndim == 3 else signal_data[0]
    Q = signal_data[..., 1, :] if signal_data.ndim == 3 else signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


# ===== 改动 4：预测函数适配新模型的输入格式 =====
def predict_single_signal(model, signal_data, device, class_list):
    """
    对单个信号进行调制识别预测。

    新模型 CNN_BiLSTM_Attention 期望输入: (batch, 1024, 2)
    原始信号数据形状: (1, 2, 1024)

    维度变换流程:
        (1, 2, 1024)  →  squeeze  →  (2, 1024)
                      →  permute  →  (1024, 2)
                      →  unsqueeze → (1, 1024, 2)  ← 这就是模型需要的
    """
    if signal_data.shape != (1, 2, 1024):
        raise ValueError(f"Expected shape (1, 2, 1024), got {signal_data.shape}")

    model.eval()

    # (1, 2, 1024) → (2, 1024) → tensor
    sample = signal_data.squeeze(0)  # (2, 1024)
    input_tensor = torch.from_numpy(sample).float()  # (2, 1024)
    input_tensor = input_tensor.permute(1, 0)  # (1024, 2) —— 与 Dataset.__getitem__ 一致
    input_tensor = input_tensor.unsqueeze(0)  # (1, 1024, 2) —— 添加 batch 维度
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)  # (1, num_classes)

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
        # ===== 改动 5：用新的 build_model 构建模型 =====
        print("Loading model architecture...")
        model = build_model(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        print("Model loaded successfully.")

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # 查找测试文件
        print(f"\nScanning for '{FILE_EXTENSION}' files in '{TEST_DATA_FOLDER}'...")
        search_pattern = os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}')
        all_files = glob.glob(search_pattern)

        if not all_files:
            print(f"Error: No '{FILE_EXTENSION}' files found in the specified folder.")
            exit()

        files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
        print(f"Found {len(all_files)} files. Will test the first {len(files_to_test)} samples.")

        # 循环预测
        prediction_counts = Counter()

        print("\n" + "=" * 50)
        print("STARTING BATCH PREDICTION")
        print("=" * 50)

        for i, file_path in enumerate(files_to_test):
            print(f"\n--- Sample {i + 1}/{len(files_to_test)}: {os.path.basename(file_path)} ---")

            try:
                # 加载 .mat 文件
                with h5py.File(file_path, 'r') as data_file:
                    single_signal_sample = data_file['data_save'][:]

                # 检查形状
                if single_signal_sample.shape != (1, 2, 1024):
                    print(f"  [Skipping] Shape: {single_signal_sample.shape}, expected (1, 2, 1024).")
                    continue

                # ===== 改动 6：使用功率归一化替代 min-max =====
                normalized_signal_sample = power_normalize(single_signal_sample)

                # 预测
                predicted_modulation, confidence = predict_single_signal(
                    model=model,
                    signal_data=normalized_signal_sample,
                    device=device,
                    class_list=classes
                )

                print(f"  -> Predicted: {predicted_modulation}")
                print(f"  -> Confidence: {confidence * 100:.2f}%")
                prediction_counts[predicted_modulation] += 1

            except FileNotFoundError:
                print(f"  [Error] File not found: '{file_path}'")
            except Exception as e:
                print(f"  [Error] {os.path.basename(file_path)}: {e}")

        # 统计摘要
        print("\n" + "=" * 50)
        print("           PREDICTION SUMMARY")
        print("=" * 50)

        total_processed = sum(prediction_counts.values())
        print(f"Total processed: {total_processed}/{len(files_to_test)}")

        if not prediction_counts:
            print("No successful predictions.")
        else:
            print("\nPrediction Counts:")
            for class_name in classes:
                count = prediction_counts.get(class_name, 0)
                pct = (count / total_processed * 100) if total_processed > 0 else 0
                print(f"  {class_name:<12}: {count:>4} times  ({pct:5.1f}%)")

            # 找出最多的预测类别
            most_common = prediction_counts.most_common(1)[0]
            print(f"\nMost predicted class: {most_common[0]} ({most_common[1]} times, "
                  f"{most_common[1] / total_processed * 100:.1f}%)")

        print("=" * 50)