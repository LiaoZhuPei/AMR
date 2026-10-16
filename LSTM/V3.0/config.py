# config.py

import torch

# --- 1. 设备和类别配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4QPSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK','64QAM','32QAM']

# #############################################################################
# 权重分配
# #############################################################################
HARD_CLASSES = ['Pi4QPSK', 'Pi4DQPSK']
# 2. 定义惩罚的倍数 (这是一个可以调整的超参数)
CONFUSION_PENALTY = 3.0  # 意味着对这两个类的错分惩罚是其他类的3倍

# 3. 自动生成权重列表
# 首先，所有类别的权重都设为 1.0
CLASS_WEIGHTS = [1.0] * len(CLASSES)
# 然后，为我们指定的困难类别设置更高的权重
for i, class_name in enumerate(CLASSES):
    if class_name in HARD_CLASSES:
        CLASS_WEIGHTS[i] = CONFUSION_PENALTY
        print(f"Applying penalty weight {CONFUSION_PENALTY} to class: {class_name} (index {i})")
# #############################################################################

# --- 2. 文件路径配置 ---
# 使用字典来组织路径，更清晰
DATA_PATHS = {
    'train_data': 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train.csv',
    'train_labels': 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train_labels.csv',
    'train_snr': 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train_snr.csv',
    'test_data': 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test.csv',
    'test_labels': 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_labels.csv',
    'test_snr': 'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_snr.csv'
}

FIGURE_PATHS = {
    'confusion_matrix': 'figure/pytorch_cnn_5features_norm_total_confusion.png'
}

WEIGHTS_PATH = 'weights/pytorch_cnn_2410_5features_normalized_DataCla12_A.pt'

# --- 3. 数据参数 ---
DATA_PARAMS = {
    'iq_channels': 2,
    'signal_length': 1024,
    'val_split_ratio': 0.2
}

# --- 4. 模型参数 ---
"""
#LSTM
MODEL_PARAMS = {
    'input_size': 5,  # 关键: I, Q, Amp, Phase, Delta_Phase
    'hidden_size': 128, 
    'num_layers': 2
}
"""
#CNN
MODEL_PARAMS = {
    'input_shape': [1, 2, 1024], # [C, H, W] for Conv2d
    # 其他参数如 hidden_size, num_layers 不再需要
}

# --- 5. 训练超参数 ---
TRAIN_PARAMS = {
    'batch_size': 400,
    'num_epochs': 500,
    'learning_rate': 0.001,
    'early_stopping_patience': 20,
    'lr_scheduler_patience': 10,
    'gradient_clip_norm': 1.0
}