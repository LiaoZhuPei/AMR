import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools  # 假设 mltools 是你自己的工具库

# 导入 PyTorch 模型
import LSTM as lstm_model

# --- 1. 设备和类别设置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classes = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

# --- 2. 数据预处理函数 (回归 Min-Max) ---
def per_sample_normalize_single(signal_data):
    """
    [回归] Min-Max 归一化。
    将单个样本缩放到 [-1, 1] 区间。
    这种方式对 OQPSK, FSK, 64QAM 等大多数信号效果最好。
    """
    # 确保是浮点型
    signal_data = signal_data.astype(np.float32)
    
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    
    if range_val == 0:
        return np.zeros_like(signal_data)
    
    # 归一化公式: 2 * (x - min) / (max - min) - 1
    normalized_data = 2 * (signal_data - min_val) / range_val - 1
    return normalized_data


# --- 3. 自定义 PyTorch Dataset (新增数据增强) ---
class IQDataset(Dataset):
    def __init__(self, iq_data, labels, apply_normalization=True, mode='train'):
        """
        Args:
            iq_data: 原始数据
            labels: 标签
            apply_normalization: 是否归一化 (通常为 True)
            mode: 'train' 表示训练模式 (开启增强), 'val'/'test' 表示评估模式 (关闭增强)
        """
        # 直接存储原始的 NumPy 数据
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.apply_normalization = apply_normalization
        self.mode = mode  # 记录当前模式
        print(f"Dataset created in '{self.mode}' mode. Augmentation: {'ON' if mode == 'train' else 'OFF'}.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. 获取原始样本 (必须 copy，防止修改原数据)
        sample = self.raw_data[idx].copy()

        # 2. 第一步：始终进行标准的 Min-Max 归一化 -> 变到 [-1, 1]
        if self.apply_normalization:
            sample = per_sample_normalize_single(sample)

        # 3. [关键增强] 第二步：仅在训练模式下，随机缩小幅度
        # 这让模型学会识别 "幅值只有 0.6 的 16QAM" (模拟缺失外圈的情况)
        if self.mode == 'train':
            # 生成 0.5 到 1.0 之间的随机因子
            scale_factor = np.random.uniform(0.5, 1.0)
            sample = sample * scale_factor

        # 4. 转 Tensor
        sample_tensor = torch.from_numpy(sample).float()

        # 5. 调整维度 (Batch, Length, Channels) -> (500, 2)
        sample_tensor = sample_tensor.permute(1, 0)

        return sample_tensor, self.labels[idx]


# --- 4. 数据加载 ---
print("Loading data from .csv files...")
IQ_CHANNELS = 2
SIGNAL_LENGTH = 500

# 请确保路径正确
train_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/PART/dataA2/train/train.csv',
    header=None
).values
num_train_samples = train_data_flat.shape[0]
train_raw = train_data_flat.reshape(num_train_samples, IQ_CHANNELS, SIGNAL_LENGTH)
print(f"Loaded {num_train_samples} training samples. Shape: {train_raw.shape}")

test_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/PART/dataA2/test/test.csv',
    header=None
).values
num_test_samples = test_data_flat.shape[0]
test_raw = test_data_flat.reshape(num_test_samples, IQ_CHANNELS, SIGNAL_LENGTH)
print(f"Loaded {num_test_samples} test samples. Shape: {test_raw.shape}")

train_labels_idx = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/PART/dataA2/train/train_labels.csv',
                               header=None).values.flatten()
test_labels_idx = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/PART/dataA2/test/test_labels.csv',
                              header=None).values.flatten()
Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]

train_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/PART/dataA2/train/train_snr.csv', header=None).values
Z_test = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/PART/dataA2/test/test_snr.csv', header=None).values

# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx, test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

# --- 修改点: 实例化 Dataset 时指定 mode ---
batch_size = 64

# 训练集：开启增强
train_dataset = IQDataset(X_train, Y_train, apply_normalization=True, mode='train')
# 验证集：关闭增强 (我们要用标准数据评估模型性能)
val_dataset = IQDataset(X_val, Y_val, apply_normalization=True, mode='val')
# 测试集：关闭增强
test_dataset = IQDataset(test_raw, test_labels_idx, apply_normalization=True, mode='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# --- 5. 模型、损失函数、优化器定义 ---
model = lstm_model.build_model(classes=len(classes)).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)


# --- 6. 训练循环 ---
nb_epoch = 500
# --- 修改文件名，避免覆盖之前的模型 ---
filepath = 'weights/pytorch_lstm_iq_minmax_aug.pt' 

if not os.path.exists('weights'):
    os.makedirs('weights')

best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 20
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(nb_epoch):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Train]")
    
    # 在这个循环中，train_loader 会自动调用 Dataset 的 __getitem__，
    # 从而触发随机幅度增强 (Scale Augmentation)
    for inputs, labels in train_pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_pbar.set_postfix({'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})
    
    avg_train_loss = train_loss / train_total
    avg_train_acc = train_correct / train_total

    # 验证阶段 (Validation) - 这里的 Loader 对应 mode='val'，没有增强
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Val]")
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_pbar.set_postfix(
                {'val_loss': f'{val_loss / val_total:.4f}', 'val_acc': f'{val_correct / val_total:.4f}'})
    
    avg_val_loss = val_loss / val_total
    avg_val_acc = val_correct / val_total

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model ...")
        torch.save(model.state_dict(), filepath)
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"EarlyStopping counter: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

print("Training finished.")
mltools.show_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])


# --- 7. 评估 ---
def predict_pytorch(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds_softmax = []
    predict_pbar = tqdm(test_loader, desc="Predicting on Test Set")
    with torch.no_grad():
        for inputs, _ in predict_pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_preds_softmax.append(probabilities.cpu().numpy())
    test_Y_hat = np.concatenate(all_preds_softmax, axis=0)
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')
    # 更新图片文件名
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename='figure/pytorch_lstm_iq_minmax_aug_confusion.png') 
    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)


print("\n--- Starting Evaluation ---")
predict_pytorch(model, test_loader, device, filepath)

print("Process finished.")