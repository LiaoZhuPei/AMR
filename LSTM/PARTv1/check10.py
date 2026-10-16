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
    'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset150/train/train.csv',
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

# 标签
# Keras的to_categorical生成one-hot编码，PyTorch的CrossEntropyLoss需要类别索引
train_labels_df = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset150/train/train_labels.csv',
                              header=None)
train_labels_idx = train_labels_df.values.flatten()

test_labels_df = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/mydateset/test/test_labels.csv', header=None)
test_labels_idx = test_labels_df.values.flatten()


# 为了和mltools中的函数兼容，我们仍然需要one-hot版本用于评估
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


Y_test_onehot = to_categorical(test_labels_idx, len(classes))

# SNR
train_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset150/train/train_snr.csv', header=None).values
test_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/mydateset/test/test_snr.csv', header=None).values
Z_test = test_snr

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++       信号可视化部分 (V3 - 为每个组合生成10个独立样本图)         +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib.pyplot as plt

print("\nStarting signal visualization process (generating up to 10 separate samples)...")

# --- 1. 设置参数和创建输出目录 ---
output_dir = 'signal_visualizations_separate_samples_dataset150'
os.makedirs(output_dir, exist_ok=True)
print(f"Separate sample plots will be saved in '{output_dir}/' directory.")
NUM_SAMPLES_TO_GENERATE = 10

# 为了可视化，我们使用重塑后但尚未标准化的原始训练数据。
# 此时 train_data 的形状是 (N, 2, 1024)
train_data_for_viz = train_data.copy()

# 获取所有唯一的SNR值
unique_snrs = np.unique(train_snr)

# --- 2. 循环遍历每个类别和每个SNR ---
for i, class_name in enumerate(tqdm(classes, desc="Processing Classes")):
    for snr_val in unique_snrs:
        # 查找所有同时满足类别和SNR条件的样本索引
        class_indices = np.where(train_labels_idx == i)[0]
        snr_indices = np.where(train_snr.flatten() == snr_val)[0]
        target_indices = np.intersect1d(class_indices, snr_indices)

        # 如果找不到任何样本，则跳过
        if len(target_indices) == 0:
            continue

        # --- 3. 为找到的前10个样本分别创建图像 ---
        # 循环处理找到的样本，最多处理 NUM_SAMPLES_TO_GENERATE 个
        # target_indices[:NUM_SAMPLES_TO_GENERATE] 会自动处理找到的样本不足10个的情况
        for sample_num, sample_index in enumerate(target_indices[:NUM_SAMPLES_TO_GENERATE]):
            # a. 提取当前循环样本的 I 和 Q 分量
            I_data = train_data_for_viz[sample_index, 0, :]
            Q_data = train_data_for_viz[sample_index, 1, :]

            # b. 绘图
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))
            # 在标题中注明这是第几个样本
            title = f'Modulation: {class_name} | SNR: {snr_val} dB | Sample #{sample_num + 1}'
            fig.suptitle(title, fontsize=16)

            # i. IQ 波形图
            ax1 = axes[0]
            ax1.plot(I_data, label='I Component', color='b')
            ax1.plot(Q_data, label='Q Component', color='r', alpha=0.7)
            ax1.set_title('IQ Waveform')
            ax1.set_xlabel('Sample Index')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlim(0, SIGNAL_LENGTH)

            # ii. 星座图
            ax2 = axes[1]
            ax2.scatter(I_data, Q_data, s=5, alpha=0.3)
            ax2.set_title('Constellation Diagram')
            ax2.set_xlabel('In-Phase (I)')
            ax2.set_ylabel('Quadrature (Q)')
            ax2.axis('equal')
            ax2.grid(True)

            # iii. 功率谱密度 (PSD)
            ax3 = axes[2]
            complex_signal = I_data + 1j * Q_data
            ax3.psd(complex_signal, NFFT=1024, Fs=1.0)
            ax3.set_title('Power Spectral Density (PSD)')

            # c. 保存图像并关闭
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # 创建包含样本编号的文件名，zfill(2)用于补零 (e.g., 01, 02, ..., 10)
            file_sample_num = str(sample_num + 1).zfill(2)
            filename = os.path.join(output_dir, f'{class_name}_{int(snr_val)}dB_sample_{file_sample_num}.png')
            plt.savefig(filename)
            plt.close(fig)  # 在循环内部关闭图像，防止内存溢出

print("\nSeparate sample visualization process finished.")
print("-" * 50)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++                  可视化部分结束                                   +++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 您脚本的后续部分会继续执行...

# 您脚本的后续部分会继续执行...

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

