import os
import numpy as np
# h5py is not used
import pandas as pd
import torch
import torch.nn.functional as F
import h5py
# 导入您自定义的CNN模型结构
import CNN as mcl

# --- 1. 基本配置 ---
"""
classes = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '64PSK',
           '4QAM', '8QAM', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
           '2FSK', '4FSK', '8FSK', '16FSK',
           '4PAM', '8PAM', '16PAM',
           'AM-DSB', 'AM-DSB-SC', 'AM-USB', 'AM-LSB', 'FM', 'PM']
"""
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', 'MSK', 'Pi4DQPSK']

MODEL_WEIGHTS_PATH = 'weights/weights0716d.pt'
TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/output_mat_files_QAM16/segment_000008.mat'
#TEST_DATA_PATH = 'E:/M_Project/P/dataset-generator/train_individual_csv_data/Pi4DQPSK_01343_32dB.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# <<<<< 重要: 填入您训练时计算出的全局均值和标准差 >>>>>
# 这些值必须与训练时使用的值完全一致
# 请从您的训练脚本输出中获取这些值并填入此处。
# 例如: TRAIN_MEAN = 0.000123, TRAIN_STD = 0.034567
TRAIN_MEAN = 0.000033  # 0.045867
TRAIN_STD = 0.181401   # 2.696623
# <<<<< 结束填写 >>>>>


# --- 2. 核心预测函数 (保持不变) ---
def predict_single_signal(model, signal_data, device, class_list):
    """
    对单个信号进行调制识别预测。
    输入信号数据形状应为 (2, 1024)。
    """
    if signal_data.shape != (1, 2, 1024):
        raise ValueError(f"Expected signal_data shape (2, 1024), but got {signal_data.shape}")

    model.eval()

    # 预处理输入数据:
    # NumPy (2, 1024) -> Tensor (2, 1024)
    # -> (1, 2, 1024) [增加批次维度]
    # -> (1, 1, 2, 1024) [增加通道维度]
    input_tensor = torch.from_numpy(signal_data).float()
    #input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)
    confidence, predicted_idx_tensor = torch.max(probabilities, 1)

    predicted_idx = predicted_idx_tensor.item()
    confidence_score = confidence.item()
    predicted_class = class_list[predicted_idx]

    return predicted_class, confidence_score


"""
try:
    with h5py.File(TEST_DATA_PATH, 'r') as data_file:
        # 直接读取数据集，因为我们知道它只包含一个样本
        single_signal_sample = data_file['data_save'][:]
    print(f"Data loaded from file with shape: {single_signal_sample.shape}")
"""

"""
        try:
            # !!!!! 修正后的数据加载逻辑 !!!!!
            # 使用pandas读取CSV文件
            # header=None 表示我们的CSV文件没有标题行
            df = pd.read_csv(TEST_DATA_PATH, header=None)

            # 将pandas DataFrame转换为NumPy数组
            single_signal_sample = df.to_numpy()
            
            if single_signal_sample.shape != (2, 1024):
                raise ValueError(f"Data shape is {single_signal_sample.shape}, not the expected (2, 1024).")


"""

# --- 3. 主执行程序 ---
if __name__ == "__main__":
    """
    if TRAIN_MEAN == 0.0 or TRAIN_STD == 1.0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 警告: 您正在使用默认的 TRAIN_MEAN 和 TRAIN_STD 值。      !!!")
        print("!!! 请务必将它们替换为训练时计算出的真实值，否则预测可能不准。!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    """
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at '{MODEL_WEIGHTS_PATH}'")
    else:
        # a. 加载模型
        print("Loading model architecture...")
        model = mcl.CNN(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)

        # b. 加载单个信号文件进行预测
        print(f"Loading a single sample from '{TEST_DATA_PATH}'...")

        try:
            with h5py.File(TEST_DATA_PATH, 'r') as data_file:
                # 直接读取数据集，因为我们知道它只包含一个样本
                single_signal_sample = data_file['data_save'][:]
            print(f"Data loaded from file with shape: {single_signal_sample.shape}")

            # <<<<< 修正后的预处理：使用全局训练统计数据进行标准化 >>>>>
            print("\n--- Pre-processing: Applying Standardization (as in training) ---")
            print(f"Using pre-calculated training stats: mean={TRAIN_MEAN:.6f}, std={TRAIN_STD:.6f}")

            # 应用与训练时完全相同的标准化公式
            normalized_signal_sample = (single_signal_sample - TRAIN_MEAN) / TRAIN_STD
            print("Standardization applied.")
            # <<<<< 修正结束 >>>>>

            print(f"\n--- Predicting Signal ---")

            # c. 调用预测函数 (使用标准化后的数据)
            predicted_modulation, confidence = predict_single_signal(
                model=model,
                signal_data=normalized_signal_sample, # <-- 传入标准化后的信号
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