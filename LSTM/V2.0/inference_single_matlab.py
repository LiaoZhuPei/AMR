# 对于matlab中仿真生成的单一类别文件

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

# 导入您的 LSTM 模型定义
import LSTM as lstm_model

# ===================================================================
# --- 1. 配置参数---
# ===================================================================

# --- 模型路径配置 ---
# 已经训练好的模型权重文件
MODEL_TO_EVALUATE_PATH = 'weights/finetuned_lstm_mode_A_11claH.pt' 

# --- 数据路径配置 ---
# 只包含单一类别的 CSV 文件
INFERENCE_DATA_PATH = 'E:/M_Project/matH32/matH32/test_11.csv' # <--- 修改这里

# --- 类别信息 ---
ALL_CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4',
               '16QAM', '2ASK', '16APSK', '32APSK', '64QAM','32QAM']


KNOWN_TRUE_CLASS = 'Pi4' # <--- 已知文件中的类别

# --- 推理超参数 ---
BATCH_SIZE = 400

# --- 设备配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# --- 2. 工具函数与数据加载器---
# ===================================================================

def per_sample_normalize_single(signal_data):
    """逐样本进行最小-最大归一化到 [-1, 1]"""
    min_val, max_val = np.min(signal_data), np.max(signal_data)
    range_val = max_val - min_val
    return np.zeros_like(signal_data) if range_val == 0 else 2 * (signal_data - min_val) / range_val - 1

class InferenceIQDataset(Dataset):
    """
    一个简化的数据集类，专门用于推理。
    它不需要标签文件。
    """
    def __init__(self, iq_data, apply_normalization=True):
        self.raw_data = iq_data
        self.apply_normalization = apply_normalization

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        if self.apply_normalization:
            sample = per_sample_normalize_single(sample)
        sample_tensor = torch.from_numpy(sample).float()
        sample_tensor = sample_tensor.permute(1, 0)
        # DataLoader 要求返回两个值，用不到标签，所以返回一个占位符 0
        return sample_tensor, 0

# ===================================================================
# --- 3. 核心推理逻辑 ---
# ===================================================================
if __name__ == "__main__":
    print(f"--- Starting Single-Class Inference ---")
    print(f"Using device: {device}")
    
    # --- 步骤 1: 加载待推理的数据 ---
    print(f"\n--- Phase 1: Loading data from '{INFERENCE_DATA_PATH}' ---")
    IQ_CHANNELS, SIGNAL_LENGTH = 2, 1024
    try:
        data_flat = pd.read_csv(INFERENCE_DATA_PATH, header=None).values
        data_raw = data_flat.reshape(-1, IQ_CHANNELS, SIGNAL_LENGTH)
        
        # 创建 Dataset 和 DataLoader
        inference_dataset = InferenceIQDataset(data_raw, apply_normalization=True)
        inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        total_samples = len(data_raw)
        print(f"Successfully loaded {total_samples} samples.")
    except FileNotFoundError:
        print(f"Error: Inference data file not found at '{INFERENCE_DATA_PATH}'.")
        print("Please check the path in the configuration section.")
        exit()

    # --- 步骤 2: 加载模型并设置为评估模式 ---
    print(f"\n--- Phase 2: Loading model from '{MODEL_TO_EVALUATE_PATH}' ---")
    if not os.path.exists(MODEL_TO_EVALUATE_PATH):
        print(f"Error: Model file not found.")
        exit()
        
    eval_model = lstm_model.build_model(classes=len(ALL_CLASSES))
    eval_model.load_state_dict(torch.load(MODEL_TO_EVALUATE_PATH, map_location=device))
    eval_model.to(device)
    eval_model.eval() # 切换到评估模式
    print("Model loaded successfully.")

    # --- 步骤 3: 执行推理 ---
    print("\n--- Phase 3: Predicting classes ---")
    all_predictions = []
    
    with torch.no_grad(): # 禁用梯度计算以加速
        for inputs, _ in tqdm(inference_loader, desc="Predicting"):
            outputs = eval_model(inputs.to(device))
            # 获取概率最高的类别的索引
            _, predicted_indices = torch.max(outputs.data, 1)
            all_predictions.extend(predicted_indices.cpu().numpy())
    
    print("Prediction finished.")

    # --- 步骤 4: 统计并展示结果 ---
    print("\n--- Phase 4: Prediction Summary ---")
    
    # 使用 Counter 统计每个类别被预测了多少次
    prediction_counts = Counter(all_predictions)
    
    print(f"Total Samples Processed: {total_samples}")
    if KNOWN_TRUE_CLASS:
        print(f"Known True Class: '{KNOWN_TRUE_CLASS}'")
    print("\nPrediction Distribution:")
    print("-" * 30)
    
    # 按预测数量从多到少排序并展示
    sorted_preds = sorted(prediction_counts.items(), key=lambda item: item[1], reverse=True)
    
    for class_index, count in sorted_preds:
        class_name = ALL_CLASSES[class_index]
        percentage = (count / total_samples) * 100
        print(f"- Predicted as '{class_name}': {count} samples ({percentage:.2f}%)")
    
    print("-" * 30)
    
    # 检查是否有类别未被预测到
    if len(prediction_counts) < len(ALL_CLASSES):
        print("\nNote: Some classes were not predicted for any sample.")

    print("\nProcess finished.")