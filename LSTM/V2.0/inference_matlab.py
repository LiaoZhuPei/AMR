# 用于推理Matlab的仿真程序，和训练代码中相同的测试集
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 确保 mltools.py 和 LSTM.py 在工作目录或 Python 路径中
import mltools
import LSTM as lstm_model

# ===================================================================
# --- 1. 配置参数 (请根据您的实际情况修改) ---
# ===================================================================

# --- 模式配置 (这会影响输出文件的命名，请与您加载的模型训练时的模式保持一致) ---
# 在您的原代码中，这个变量是 FINETUNE_MODE
# 例如，如果您加载的是 'mode_A' 训练出的模型，这里就设置为 'A'
MODEL_TRAINING_MODE = 'A'  # <--- 在这里选择 'A' 或 'B'

# --- 路径配置 ---
# 【需要加载的模型】: 指向您已经通过迁移学习训练好的模型权重文件
MODEL_TO_EVALUATE_PATH = f'weights/finetuned_lstm_mode_{MODEL_TRAINING_MODE}.pt'

# 【需要测试的数据集】
TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test.csv'
TEST_LABELS_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_labels.csv'
TEST_SNR_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_snr.csv'

# 【结果输出路径】: 推理结果（如混淆矩阵图片）的保存位置
OUTPUT_DIR = 'figure'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 类别配置 ---
# 【重要】: 这里的类别列表必须与您加载的模型训练时所用的类别完全一致
ALL_CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4QPSK',
               '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK', '64QAM','32QAM']

# --- 推理超参数 ---
BATCH_SIZE = 400

# --- 设备配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# --- 2. 工具函数与数据加载器 (从原脚本复用) ---
# ===================================================================

def per_sample_normalize_single(signal_data):
    """逐样本进行最小-最大归一化到 [-1, 1]"""
    min_val, max_val = np.min(signal_data), np.max(signal_data)
    range_val = max_val - min_val
    return np.zeros_like(signal_data) if range_val == 0 else 2 * (signal_data - min_val) / range_val - 1

class IQDataset(Dataset):
    """
    用于加载 I/Q 信号数据的数据集类。
    在推理时，我们不应用数据增强 (is_train=False)。
    """
    def __init__(self, iq_data, labels, apply_normalization=True, is_train=False):
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.apply_normalization = apply_normalization
        # 在推理脚本中，is_train 应始终为 False，以确保数据一致性
        if is_train:
            print("Warning: Data augmentation should be disabled during inference. Forcing is_train=False.")
            self.is_train = False
        else:
            self.is_train = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. 获取原始的单个样本 (shape: 2, 1024)
        sample = self.raw_data[idx]

        # 2. 如果需要，应用逐样本归一化
        if self.apply_normalization:
            sample = per_sample_normalize_single(sample)

        # 3. 将处理后的 NumPy 数组转换为 PyTorch Tensor
        sample_tensor = torch.from_numpy(sample).float()

        # 4. 调整维度以匹配 LSTM 输入 (N, L, C) -> (1024, 2)
        sample_tensor = sample_tensor.permute(1, 0)

        return sample_tensor, self.labels[idx]

# ===================================================================
# --- 3. 核心推理逻辑 ---
# ===================================================================
if __name__ == "__main__":
    print(f"--- Starting Inference Process ---")
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_TO_EVALUATE_PATH}")

    # --- 步骤 1: 加载待评估的测试数据 ---
    print("\n--- Phase 1: Loading Test Data ---")
    IQ_CHANNELS, SIGNAL_LENGTH = 2, 1024
    try:
        test_data_flat = pd.read_csv(TEST_DATA_PATH, header=None).values
        test_raw = test_data_flat.reshape(-1, IQ_CHANNELS, SIGNAL_LENGTH)
        test_labels_idx = pd.read_csv(TEST_LABELS_PATH, header=None).values.flatten()
        # 将标签转换为 one-hot 编码，用于 mltools
        Y_test_onehot = np.eye(len(ALL_CLASSES), dtype='uint8')[test_labels_idx]
        # 加载信噪比数据
        Z_test = pd.read_csv(TEST_SNR_PATH, header=None).values

        # 创建 Dataset 和 DataLoader
        test_dataset = IQDataset(test_raw, test_labels_idx, apply_normalization=True, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"Successfully loaded {len(test_raw)} test samples.")
    except FileNotFoundError as e:
        print(f"Error: Test data file not found. {e}")
        print("Please check the paths in the configuration section.")
        exit()

    # --- 步骤 2: 加载模型并设置为评估模式 ---
    print("\n--- Phase 2: Model Preparation ---")
    if not os.path.exists(MODEL_TO_EVALUATE_PATH):
        print(f"Error: Model file not found at '{MODEL_TO_EVALUATE_PATH}'.")
        exit()
        
    # 实例化模型结构，输出类别数必须与 ALL_CLASSES 长度一致
    eval_model = lstm_model.build_model(classes=len(ALL_CLASSES))
    # 加载已经训练好的权重
    eval_model.load_state_dict(torch.load(MODEL_TO_EVALUATE_PATH, map_location=device))
    eval_model.to(device)
    # 切换到评估模式 (这会关闭 Dropout 等层)
    eval_model.eval()
    print("Model loaded and set to evaluation mode.")

    # --- 步骤 3: 执行推理 ---
    print("\n--- Phase 3: Predicting on Test Set ---")
    all_preds_softmax = []
    predict_pbar = tqdm(test_loader, desc="Predicting")

    # 使用 torch.no_grad() 来禁用梯度计算，节省内存并加速
    with torch.no_grad():
        for inputs, _ in predict_pbar:
            # 将数据移动到指定设备
            inputs = inputs.to(device)
            # 模型前向传播
            outputs = eval_model(inputs)
            # 应用 softmax 获取概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # 将结果从 GPU 移回 CPU 并转换为 numpy 数组
            all_preds_softmax.append(probabilities.cpu().numpy())
    
    # 将所有批次的预测结果合并成一个大的 numpy 数组
    test_Y_hat = np.concatenate(all_preds_softmax, axis=0)
    print("Prediction finished.")

    # --- 步骤 4: 评估与可视化 ---
    print("\n--- Phase 4: Calculating Metrics and Visualization ---")
    # 1. 计算总体准确率和混淆矩阵
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, ALL_CLASSES)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy on Test Set: {acc * 100:.2f}% ({right}/{right + wrong})')

    # 2. 绘制并保存混淆矩阵
    conf_matrix_filename = os.path.join(OUTPUT_DIR, f'inference_mode_{MODEL_TRAINING_MODE}_confusion.png')
    mltools.plot_confusion_matrix(cm, labels=ALL_CLASSES, save_filename=conf_matrix_filename)
    print(f"Confusion matrix saved to '{conf_matrix_filename}'")

    # 3. 按信噪比计算准确率 (如果提供了信噪比数据)
    if Z_test is not None:
        print("\n--- Accuracy per SNR ---")
        mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, ALL_CLASSES, min_snr=-18)

    print("\nProcess finished.")