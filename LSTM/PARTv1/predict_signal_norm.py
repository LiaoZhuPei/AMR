import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
# 导入您自定义的CNN模型结构
import CNN as mcl

# --- 1. 基本配置 ---
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', 'MSK', 'Pi4DQPSK']

##### 修改点 1: 更新模型权重路径 #####
# 确保这个路径指向你用 "逐样本归一化" 方法训练保存的模型
MODEL_WEIGHTS_PATH = 'weights/weights_persample_norm2.pt'

TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/output_mat_files_OQPSK/segment_000088.mat'
# TEST_DATA_PATH = 'E:/M_Project/P/dataset-generator/train_individual_csv_data/Pi4DQPSK_01343_32dB.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


##### 修改点 2: 移除全局均值和标准差 #####
# 由于我们现在使用逐样本归一化，不再需要全局的 TRAIN_MEAN 和 TRAIN_STD
# TRAIN_MEAN = 0.000033
# TRAIN_STD = 0.181401
# ------------------------------------

# --- 2. 核心函数 ---

##### 修改点 3: 添加逐样本归一化函数 #####
# 这个函数应该与你训练脚本中的函数完全一致
def per_sample_normalize_single(signal_data):
    """
    对单个样本（独立的时间序列）进行归一化，将其缩放到 [-1, 1] 区间。
    输入 shape: (2, 1024) 或 (1, 2, 1024)
    输出 shape: 与输入相同
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)

    range_val = max_val - min_val

    if range_val == 0:
        # 如果信号是常数，返回全零数组，形状与输入一致
        return np.zeros_like(signal_data)

    normalized_data = 2 * (signal_data - min_val) / range_val - 1
    return normalized_data


# ------------------------------------

def predict_single_signal(model, signal_data, device, class_list):
    """
    对单个信号进行调制识别预测。
    输入信号数据形状应为 (2, 1024)。
    """
    # 形状检查应在预处理之后进行，或者确保输入就是 (2, 1024)
    if signal_data.shape != (1, 2, 1024):
        raise ValueError(f"Expected signal_data shape (2, 1024), but got {signal_data.shape}")

    model.eval()

    # 预处理输入数据:
    # NumPy (2, 1024) -> (1, 2, 1024) [增加批次维度] -> (1, 1, 2, 1024) [增加通道维度]
    input_tensor = torch.from_numpy(signal_data).float()
    #input_tensor = input_tensor.unsqueeze(0)  # 增加批次维度
    input_tensor = input_tensor.unsqueeze(0).to(device)  # 增加通道维度并送到设备

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
        # a. 加载模型
        print("Loading model architecture...")
        # 假设你的 mcl.build_model() 函数就是返回 CNN 模型实例
        model = mcl.build_model(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)

        # b. 加载单个信号文件进行预测
        print(f"Loading a single sample from '{TEST_DATA_PATH}'...")

        try:
            # 数据加载逻辑 (根据你的文件类型选择)
            # 适用于 .mat 文件
            with h5py.File(TEST_DATA_PATH, 'r') as data_file:
                single_signal_sample = data_file['data_save'][:]
            print(f"Data loaded from .mat file with shape: {single_signal_sample.shape}")

            # 适用于 .csv 文件 (如果需要)
            # df = pd.read_csv(TEST_DATA_PATH, header=None)
            # single_signal_sample = df.to_numpy()
            # print(f"Data loaded from .csv file with shape: {single_signal_sample.shape}")

            # 确保数据形状是 (2, 1024)
            if single_signal_sample.shape != (1, 2, 1024):
                raise ValueError(
                    f"Loaded data shape is {single_signal_sample.shape}, but (2, 1024) is required for prediction.")

            ##### 修改点 4: 应用逐样本归一化 #####
            print("\n--- Pre-processing: Applying Per-Sample Normalization (as in training) ---")

            # 应用与训练时完全相同的逐样本归一化
            normalized_signal_sample = per_sample_normalize_single(single_signal_sample)

            print("Per-sample normalization applied.")
            # ------------------------------------

            print(f"\n--- Predicting Signal ---")

            # c. 调用预测函数 (使用归一化后的数据)
            predicted_modulation, confidence = predict_single_signal(
                model=model,
                signal_data=normalized_signal_sample,  # <-- 传入归一化后的信号
                device=device,
                class_list=classes
            )

            # d. 打印结果
            print(f"\nPredicted Modulation: {predicted_modulation}")
            print(f"Confidence: {confidence * 100:.2f}%")

        except FileNotFoundError:
            print(f"Error: Test data file not found at '{TEST_DATA_PATH}'")
        except Exception as e:
            print(f"An error occurred: {e}")