import os
import numpy as np
import pandas as pd
# import h5py  # 不再需要 h5py 来读取数据
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools
import CNN as mcl

# --- 设备设置 (不变) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 数据加载和预处理 (这是主要修改区域) ---

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', 'MSK', 'Pi4DQPSK']

# 定义数据参数，这对于从CSV重塑回原始形状至关重要
IQ_CHANNELS = 2
SIGNAL_LENGTH = 1024

# 训练数据
print("Loading training data...")
train_data_flat = pd.read_csv(
    'E:/M_Project/mat4/mat4/train.csv',
    header=None
).values

# !! 关键步骤：将展平的数据重塑回 (N, 2, L) 的形状 !!
# train_data_flat 的形状是 (N, 2 * 1024)
# 我们需要将其变回 (N, 2, 1024)
num_train_samples = train_data_flat.shape[0]
train_data = train_data_flat.reshape(num_train_samples, IQ_CHANNELS, SIGNAL_LENGTH)


# 测试数据
print("Loading test data from CSV...")
test_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/mydateset/test/test.csv',
    header=None
).values

# 重塑测试数据
num_test_samples = test_data_flat.shape[0]
test_data = test_data_flat.reshape(num_test_samples, IQ_CHANNELS, SIGNAL_LENGTH)

# PyTorch需要 [N, C, H, W] 格式, 当前为 [N, H, W]
# 我们需要增加一个通道维度 C=1
#train_data = np.expand_dims(train_data, axis=1)  # Shape: [N, 1, 2, 1024]
test_data = np.expand_dims(test_data, axis=1)  # Shape: [N, 1, 2, 1024]
print("Data shapes adjusted for PyTorch.")
"""
# ++++++++++++++++ 新增：数据标准化 ++++++++++++++++
# 1. 只使用训练数据计算全局的均值和标准差
print("Normalizing data...")
train_mean = np.mean(train_data)
train_std = np.std(train_data)
print(f"Calculated from training data: mean={train_mean:.6f}, std={train_std:.6f}")

# 2. 使用计算出的均值和标准差来标准化训练数据和测试数据
train_data = (train_data - train_mean) / train_std
test_data = (test_data - train_mean) / train_std
print("Training and test data have been normalized.")
"""
# 标签
# Keras的to_categorical生成one-hot编码，PyTorch的CrossEntropyLoss需要类别索引
train_labels_df = pd.read_csv('/M_Project/mat4/mat4/train_labels.csv',
                              header=None)
train_labels_idx = train_labels_df.values.flatten()

test_labels_df = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/mydateset/test/test_labels.csv', header=None)
test_labels_idx = test_labels_df.values.flatten()


# 为了和mltools中的函数兼容，我们仍然需要one-hot版本用于评估
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


Y_test_onehot = to_categorical(test_labels_idx, len(classes))

# SNR
train_snr = pd.read_csv('/M_Project/mat4/mat4/train_snr.csv', header=None).values
test_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/mydateset/test/test_snr.csv', header=None).values
Z_test = test_snr

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++                  信号可视化部分 (新增)                           +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib.pyplot as plt

print("\nStarting signal visualization process...")

# --- 1. 设置参数和创建输出目录 ---
output_dir = 'signal_visualizations_big'
os.makedirs(output_dir, exist_ok=True)
print(f"Visualization plots will be saved in '{output_dir}/' directory.")

# 为了可视化，我们使用重塑后但尚未标准化的原始训练数据
# train_data_original 的形状应为 (N, 1, 2, 1024)
train_data_original = np.expand_dims(train_data.copy(), axis=1)

# 获取所有唯一的SNR值
unique_snrs = np.unique(train_snr)

# --- 2. 循环遍历每个类别和每个SNR，生成并保存图像 ---
for i, class_name in enumerate(tqdm(classes, desc="Processing Classes")):
    for snr_val in unique_snrs:
        # 查找同时满足类别和SNR条件的样本索引
        # a. 找到所有属于当前类别的索引
        class_indices = np.where(train_labels_idx == i)[0]
        # b. 找到所有属于当前SNR的索引
        snr_indices = np.where(train_snr.flatten() == snr_val)[0]
        # c. 求交集，找到同时满足两个条件的样本索引
        target_indices = np.intersect1d(class_indices, snr_indices)

        # 如果找不到匹配的样本，则跳过
        if len(target_indices) == 0:
            # print(f"Warning: No sample found for {class_name} at {snr_val} dB. Skipping.")
            continue

        # 选择第一个匹配的样本进行可视化
        sample_index = target_indices[0]

        # 提取该样本的信号数据，形状为 (1, 2, 1024)
        signal_sample = train_data_original[sample_index]

        # 提取 I 和 Q 分量
        I_data = signal_sample[0, 0, :]  # I分量
        Q_data = signal_sample[0, 1, :]  # Q分量

        # --- 3. 绘图 ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        fig.suptitle(f'Modulation: {class_name}  |  SNR: {snr_val} dB', fontsize=16)

        # a. IQ 波形图
        ax1 = axes[0]
        ax1.plot(I_data, label='I Component', color='b')
        ax1.plot(Q_data, label='Q Component', color='r', alpha=0.7)
        ax1.set_title('IQ Waveform')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(0, SIGNAL_LENGTH)

        # b. 星座图
        ax2 = axes[1]
        ax2.scatter(I_data, Q_data, s=5, alpha=0.5)
        ax2.set_title('Constellation Diagram')
        ax2.set_xlabel('In-Phase (I)')
        ax2.set_ylabel('Quadrature (Q)')
        ax2.axis('equal')  # 保证星座图比例正确
        ax2.grid(True)

        # c. 功率谱密度 (PSD)
        ax3 = axes[2]
        # 将I/Q组合成复数信号
        complex_signal = I_data + 1j * Q_data
        ax3.psd(complex_signal, NFFT=1024, Fs=1.0)  # Fs=1.0 表示归一化频率
        ax3.set_title('Power Spectral Density (PSD)')
        # psd函数会自动添加xlabel和ylabel

        # 调整布局并保存图像
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 留出空间给suptitle

        # 创建文件名并保存
        filename = os.path.join(output_dir, f'{class_name}_{int(snr_val)}dB.png')
        plt.savefig(filename)
        plt.close(fig)  # 关闭图像，释放内存，非常重要！

print("\nSignal visualization process finished.")
print("-" * 50)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++                  可视化部分结束                                   +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# 划分训练集和验证集 (这是您原来的代码，紧随其后)
print("Splitting data for training...")
X_train, X_val, Y_train_idx, Y_val_idx = train_test_split(
    train_data, train_labels_idx, test_size=0.2, random_state=42
)
# ... 后续代码不变

# 划分训练集和验证集
X_train, X_val, Y_train_idx, Y_val_idx = train_test_split(
    train_data, train_labels_idx, test_size=0.2, random_state=42
)
print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

