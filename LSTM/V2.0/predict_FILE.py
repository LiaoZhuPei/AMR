
import os
import glob
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
# --- 修改点 1: 导入正确的 LSTM 模型结构 ---
import LSTM as lstm  # 之前是 CNN2，现在改为 LSTM

# --- 1. 基本配置 ---
# 确保这里的类别与你训练 LSTM 模型时使用的类别完全一致
# 根据你的训练代码，你可能使用了 '16QAM', '32QAM', '64QAM'
# 请在这里填写正确的类别列表
# classes = ['16QAM', '32QAM', '64QAM']
# 或者你使用了更全的类别列表，请确保一致
#classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
#           '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK']
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4QPSK',
               '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK', '64QAM','32QAM']

# --- 修改点 2: 更新模型和数据路径 ---
# 使用你的 LSTM 模型权重路径
MODEL_WEIGHTS_PATH = 'weights/finetuned_lstm_mode_A.pt'  # <--- 请替换为你的 LSTM 权重路径

# 指定包含测试文件的文件夹路径
TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_PI4DQPSK_raw_amplitude/'  # <--- 这是你要测试的数据
# 指定要测试的样本数量 (N)
NUM_SAMPLES_TO_TEST = 100
# 指定文件扩展名
FILE_EXTENSION = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. 核心函数 ---

# --- 修改点 3: 从 LSTM 训练代码中复制数据预处理函数 ---
# 这个函数是关键，它定义了 LSTM 模型期望的输入格式
def to_amp_phase_torch(iq_data):
    """
    使用 PyTorch 操作将 I/Q 数据转换为幅度和相位。
    输入 shape: (N, 2, 1024)
    输出 shape: (N, 1024, 2)
    """
    # 确保输入是 PyTorch 张量
    if not isinstance(iq_data, torch.Tensor):
        iq_data = torch.from_numpy(iq_data).float()

    # 创建复数张量
    complex_data = torch.complex(iq_data[:, 0, :], iq_data[:, 1, :])

    # 计算幅度和相位
    amp = torch.abs(complex_data)
    phase = torch.angle(complex_data) / torch.pi

    # 对每个样本的幅度进行最大值归一化
    amp_max = amp.max(dim=1, keepdim=True).values
    # 加上一个很小的数 epsilon 防止除以零
    amp_normalized = amp / (amp_max + 1e-8)

    # 拼接归一化后的幅度和相位
    amp_phase_data = torch.cat(
        (amp_normalized.unsqueeze(-1), phase.unsqueeze(-1)),
        dim=-1
    )
    return amp_phase_data


# --- 修改点 4: 适配 LSTM 模型的预测函数 ---
def predict_single_signal(model, signal_tensor, device, class_list):
    """
    对单个经过预处理的信号张量进行预测。
    输入信号张量形状应为 (1, 1024, 2)，且应为 PyTorch Tensor。
    """
    if not isinstance(signal_tensor, torch.Tensor):
        raise TypeError(f"Expected signal_tensor to be a PyTorch Tensor, but got {type(signal_tensor)}")

    if signal_tensor.shape != (1, 1024, 2):
        raise ValueError(f"Expected signal_tensor shape (1, 1024, 2), but got {signal_tensor.shape}")

    model.eval()

    # 将 Tensor 移动到正确的设备
    input_tensor = signal_tensor.to(device)

    # LSTM 模型通常直接接收 (batch, seq_len, features) 形状的输入
    # 所以不需要像 CNN 那样 unsqueeze(0) 来增加一个通道维度
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
        # a. 加载 LSTM 模型
        print("Loading LSTM model architecture...")
        # 使用你导入的 lstm 模块来构建模型
        model = lstm.build_model(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        print("Model loaded successfully.")

        # b. 查找并准备测试文件 (逻辑不变)
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
        print("STARTING BATCH PREDICTION (LSTM)")
        print("=" * 50)

        for i, file_path in enumerate(files_to_test):
            print(f"\n--- Processing Sample {i + 1}/{len(files_to_test)}: {os.path.basename(file_path)} ---")

            try:
                # 1. 加载单个信号文件 (逻辑不变)
                with h5py.File(file_path, 'r') as data_file:
                    single_signal_sample = data_file['data_save'][:]

                # 2. 检查数据形状 (逻辑不变)
                if single_signal_sample.shape != (1, 2, 1024):
                    print(f"  [Skipping] Incorrect data shape: {single_signal_sample.shape}. Expected (1, 2, 1024).")
                    continue

                # --- 修改点 5: 应用正确的 LSTM 预处理 ---
                # 将 NumPy 数组转换为幅度和相位格式的 PyTorch Tensor
                preprocessed_tensor = to_amp_phase_torch(single_signal_sample)

                # 4. 调用为 LSTM 适配的预测函数
                predicted_modulation, confidence = predict_single_signal(
                    model=model,
                    signal_tensor=preprocessed_tensor,  # 传递 Tensor
                    device=device,
                    class_list=classes
                )

                # 5. 打印结果和更新计数 (逻辑不变)
                print(f"  -> Predicted Modulation: {predicted_modulation}")
                print(f"  -> Confidence: {confidence * 100:.2f}%")
                prediction_counts[predicted_modulation] += 1

            except FileNotFoundError:
                print(f"  [Error] Test data file not found at '{file_path}'")
            except Exception as e:
                print(f"  [Error] An error occurred while processing {os.path.basename(file_path)}: {e}")

        # d. 打印最终的统计摘要 (逻辑不变)
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