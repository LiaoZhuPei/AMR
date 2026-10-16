import os
import glob
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt

# 导入与训练时相同的模型结构
import LSTM as lstm_model
# 导入与训练时相同的配置文件，以确保所有参数一致
import config

# --- 1. 基本配置 (从 config.py 导入) ---
# 使用与训练时完全一致的类别列表
classes = config.CLASSES
# 模型权重路径
MODEL_WEIGHTS_PATH = config.WEIGHTS_PATH
# 设备
device = config.DEVICE

# --- 本地预测配置 ---
TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_2FSK/'  # <--- 请替换为你要测试的数据文件夹
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION = '.mat'

print(f"Using device: {device}")
print(f"Model path: {MODEL_WEIGHTS_PATH}")
print(f"Test data folder: {TEST_DATA_FOLDER}")


# #############################################################################
# --- 修改点 1: 实现与训练时完全一致的预处理函数 ---
# #############################################################################
def _per_sample_normalize_single(signal_data):
    """
    私有辅助函数：对单个样本进行归一化。
    与 dataset.py 中的函数完全相同。
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data)
    return 2 * (signal_data - min_val) / range_val - 1


def preprocess_single_signal(iq_sample_raw, normalize_iq_amp=True):
    """
    对单个原始 I/Q 信号样本进行完整的5特征预处理。
    这个函数是 IQFeatureDataset.__getitem__ 的预测版本。
    输入 shape: (2, 1024) NumPy array
    输出 shape: (1, 1024, 5) PyTorch Tensor
    """
    # 1. 计算幅度和相位
    i_data_raw = iq_sample_raw[0, :]
    q_data_raw = iq_sample_raw[1, :]
    amplitude_raw = np.sqrt(i_data_raw ** 2 + q_data_raw ** 2)
    phase = np.arctan2(q_data_raw, i_data_raw)

    # 2. 计算差分相位
    delta_phase = np.diff(phase, prepend=phase[0])
    delta_phase[delta_phase > np.pi] -= 2 * np.pi
    delta_phase[delta_phase < -np.pi] += 2 * np.pi

    # 3. 归一化 I/Q 和 Amplitude
    if normalize_iq_amp:
        iq_normalized = _per_sample_normalize_single(iq_sample_raw)
        i_data_norm = iq_normalized[0, :]
        q_data_norm = iq_normalized[1, :]
        amplitude_norm = _per_sample_normalize_single(amplitude_raw)
    else:
        i_data_norm, q_data_norm, amplitude_norm = i_data_raw, q_data_raw, amplitude_raw

    # 4. 组合所有特征
    features = np.stack([
        i_data_norm, q_data_norm, amplitude_norm, phase, delta_phase
    ], axis=0)

    # 5. 转换为 PyTorch Tensor 并调整维度
    features_tensor = torch.from_numpy(features).float().permute(1, 0)  # -> (1024, 5)

    # 6. 增加一个 batch 维度以匹配模型输入
    return features_tensor.unsqueeze(0)  # -> (1, 1024, 5)


# #############################################################################
# --- 修改点 2: 调整预测函数以适应新的输入形状 ---
# #############################################################################
def predict_single_signal(model, signal_tensor, device, class_list):
    """
    对单个经过预处理的5特征信号张量进行预测。
    输入信号张量形状应为 (1, 1024, 5)。
    """
    if signal_tensor.shape != (1, 1024, 5):
        raise ValueError(f"Expected signal_tensor shape (1, 1024, 5), but got {signal_tensor.shape}")

    model.eval()
    input_tensor = signal_tensor.to(device)

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
        # #############################################################################
        # --- 修改点 3: 使用 config 文件中的参数加载模型 ---
        # #############################################################################
        print("Loading LSTM model architecture...")
        model = lstm_model.build_model(
            input_size=config.MODEL_PARAMS['input_size'],
            hidden_size=config.MODEL_PARAMS['hidden_size'],
            num_layers=config.MODEL_PARAMS['num_layers'],
            classes=len(config.CLASSES)
        )
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
        confidences = {cls: [] for cls in classes}

        print("\n" + "=" * 50)
        print("STARTING BATCH PREDICTION (5-Feature LSTM)")
        print("=" * 50)

        for i, file_path in enumerate(files_to_test):
            try:
                # 1. 加载单个信号文件
                with h5py.File(file_path, 'r') as data_file:
                    # h5py 读取的数据形状通常是 (1, 2, 1024)，我们需要去掉批次维度
                    single_signal_sample = data_file['data_save'][:].squeeze(0)

                # 2. 检查数据形状
                if single_signal_sample.shape != (2, 1024):
                    print(
                        f"  [Skipping] Incorrect data shape after squeeze: {single_signal_sample.shape}. Expected (2, 1024).")
                    continue

                # #############################################################################
                # --- 修改点 4: 应用新的、正确的预处理函数 ---
                # #############################################################################
                preprocessed_tensor = preprocess_single_signal(single_signal_sample)

                # 4. 调用预测函数
                predicted_modulation, confidence = predict_single_signal(
                    model=model,
                    signal_tensor=preprocessed_tensor,
                    device=device,
                    class_list=classes
                )

                # 5. 打印结果和更新计数
                print(f"--- Sample {i + 1}/{len(files_to_test)}: {os.path.basename(file_path)} ---")
                print(f"  -> Predicted Modulation: {predicted_modulation} (Confidence: {confidence * 100:.2f}%)")
                prediction_counts[predicted_modulation] += 1
                confidences[predicted_modulation].append(confidence)

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
            print("\nPrediction Counts & Average Confidence:")
            sorted_classes = sorted(prediction_counts.keys(), key=lambda x: prediction_counts[x], reverse=True)
            for class_name in sorted_classes:
                count = prediction_counts[class_name]
                avg_conf = np.mean(confidences[class_name]) * 100 if confidences[class_name] else 0
                print(f"- {class_name:<10}: {count:<5} times (Avg. Confidence: {avg_conf:.2f}%)")

            # 可视化预测结果的分布
            plt.figure(figsize=(10, 6))
            plt.bar(prediction_counts.keys(), prediction_counts.values())
            plt.title('Prediction Distribution')
            plt.xlabel('Modulation Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('figure/prediction_distribution.png')
            print("\nPrediction distribution plot saved to 'figure/prediction_distribution.png'")

        print("=" * 50)