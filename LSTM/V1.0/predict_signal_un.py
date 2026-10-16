import os
import numpy as np
# import h5py  <-- 移除此行
import pandas as pd  # <-- 添加此行
import torch
import torch.nn.functional as F

# 导入您自定义的CNN模型结构
import CNN as mcl

# --- 1. 基本配置 (修改了TEST_DATA_PATH) ---
classes = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '64PSK',
           '4QAM', '8QAM', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
           '2FSK', '4FSK', '8FSK', '16FSK',
           '4PAM', '8PAM', '16PAM',
           'AM-DSB', 'AM-DSB-SC', 'AM-USB', 'AM-LSB', 'FM', 'PM']
MODEL_WEIGHTS_PATH = 'weights/weights3.pt'
# 将测试文件路径指向.csv文件
TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/Pytorch2/QPSK2_00011_18dB.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. 核心预测函数 (保持不变) ---
def predict_single_signal(model, signal_data, device, class_list):
    """
    对单个信号进行调制识别预测。
    输入信号数据形状应为 (2, 1024)。
    """
    if signal_data.shape != (2, 1024):
        raise ValueError(f"Expected signal_data shape (2, 1024), but got {signal_data.shape}")

    model.eval()

    # 预处理输入数据:
    # NumPy (2, 1024) -> Tensor (2, 1024)
    # -> (1, 2, 1024) [增加批次维度]
    # -> (1, 1, 2, 1024) [增加通道维度]
    input_tensor = torch.from_numpy(signal_data).float()
    input_tensor = input_tensor.unsqueeze(0)
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
        # a. 加载模型
        print("Loading model architecture...")
        model = mcl.CNN(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)

        # b. 加载单个信号文件进行预测
        print(f"Loading a single sample from '{TEST_DATA_PATH}'...")
        try:
            # !!!!! 修正后的数据加载逻辑 !!!!!
            # 使用pandas读取CSV文件
            # header=None 表示我们的CSV文件没有标题行
            df = pd.read_csv(TEST_DATA_PATH, header=None)

            # 将pandas DataFrame转换为NumPy数组
            single_signal_sample = df.to_numpy()

            print(f"Data loaded from file with shape: {single_signal_sample.shape}")

            # 确认最终形状是否是我们期望的 (2, 1024)
            if single_signal_sample.shape != (2, 1024):
                raise ValueError(f"Data shape is {single_signal_sample.shape}, not the expected (2, 1024).")

            print(f"\n--- Predicting Signal ---")

            # c. 调用预测函数
            predicted_modulation, confidence = predict_single_signal(
                model=model,
                signal_data=single_signal_sample,
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