import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py
import mltools  # 假设 mltools 是你自己的工具库

# 导入我们新创建的PyTorch模型
import LSTM as lstm_model

# --- 1. 设备和类别设置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classes = ['16QAM', '32QAM', '64QAM']


# --- 2. 数据预处理函数 (PyTorch版本) ---
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
    complex_data = torch.complex(iq_data[:, 0, :], iq_data[:, 1, :])  # Shape: (N, 1024)

    # 计算幅度和相位
    amp = torch.abs(complex_data)
    phase = torch.angle(complex_data) / torch.pi  # PyTorch angle 直接返回弧度, 除以pi归一化

    # 对每个样本的幅度进行最大值归一化
    # amp 的 shape 是 (N, 1024)
    # amp.max(dim=1, keepdim=True) 会计算每行的最大值，并保持维度为 (N, 1)
    # 这样可以利用广播机制进行除法
    amp_max = amp.max(dim=1, keepdim=True).values

    # 防止除以零（如果一个样本的幅度全为0）
    amp_normalized = amp / torch.where(amp_max == 0, torch.tensor(1.0), amp_max)
    # --------------------------

    # 拼接归一化后的幅度和相位
    amp_phase_data = torch.cat(
        (amp_normalized.unsqueeze(-1), phase.unsqueeze(-1)),
        dim=-1
    )

    return amp_phase_data


# --- 3. 自定义 PyTorch Dataset ---
class IQDataset(Dataset):
    def __init__(self, iq_data, labels, use_amp_phase=True):
        self.use_amp_phase = use_amp_phase

        if self.use_amp_phase:
            print("Converting data to Amplitude/Phase format...")
            self.data = to_amp_phase_torch(iq_data)
        else:
            # 如果用I/Q, 调整维度为 (N, 1024, 2)
            self.data = torch.from_numpy(iq_data).float().permute(0, 2, 1)

        self.labels = torch.from_numpy(labels).long()
        print("Dataset created.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# --- 4. 数据加载 ---
print("Loading data from .csv files...")
# --- 定义信号参数，用于重塑形状 ---
IQ_CHANNELS = 2
SIGNAL_LENGTH = 1024

# --- 加载训练数据 ---
# 使用 pandas 读取 csv 文件
train_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/train/train.csv',
    header=None  # 假设你的csv文件没有表头
).values

# 获取样本数量
num_train_samples = train_data_flat.shape[0]

# 将展平的数据重塑为 (N, C, L) -> (N, 2, 1024)
# 这是关键的一步，它将二维数据恢复为三维的信号结构
train_raw = train_data_flat.reshape(num_train_samples, IQ_CHANNELS, SIGNAL_LENGTH)

print(f"Loaded {num_train_samples} training samples. Shape: {train_raw.shape}")


# --- 加载测试数据 ---
test_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/test/test.csv',
    header=None
).values

num_test_samples = test_data_flat.shape[0]

# 同样地，重塑测试数据
test_raw = test_data_flat.reshape(num_test_samples, IQ_CHANNELS, SIGNAL_LENGTH)

print(f"Loaded {num_test_samples} test samples. Shape: {test_raw.shape}")

train_labels_idx = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/train/train_labels.csv',
                               header=None).values.flatten()
test_labels_idx = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/test/test_labels.csv',
                              header=None).values.flatten()
Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]  # 用于最终评估

train_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/train/train_snr.csv', header=None).values
Z_test = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam7/test/test_snr.csv', header=None).values

# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx, test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

# 创建 Dataset 和 DataLoader
batch_size = 400
train_dataset = IQDataset(X_train, Y_train, use_amp_phase=False)
val_dataset = IQDataset(X_val, Y_val, use_amp_phase=False)
# 创建测试集时，标签使用原始索引即可
test_dataset = IQDataset(test_raw, test_labels_idx, use_amp_phase=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# --- 5. 模型、损失函数、优化器定义 ---
model = lstm_model.build_model(classes=len(classes)).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Keras Adam 默认 lr=0.001
# 等效于 Keras ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

# --- 6. 训练循环 ---
nb_epoch = 500  # 同样设置一个大值
filepath = 'weights/pytorch_lstm_weights.pt'
if not os.path.exists('weights'):
    os.makedirs('weights')

best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 60  # 等效于 Keras EarlyStopping patience=30
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(nb_epoch):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Train]")
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

    # 学习率调度
    scheduler.step(avg_val_loss)

    # 模型保存 (等效于 ModelCheckpoint)
    if avg_val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model ...")
        torch.save(model.state_dict(), filepath)
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        # 早停逻辑 (等效于 EarlyStopping)
        patience_counter += 1
        print(f"EarlyStopping counter: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

print("Training finished.")
mltools.show_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])


# --- 7. 评估 ---
def predict_pytorch(model, test_loader, device, model_path):
    # 加载最佳模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds_softmax = []
    predict_pbar = tqdm(test_loader, desc="Predicting on Test Set")
    with torch.no_grad():
        for inputs, _ in predict_pbar:  # 测试集标签不需要
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Keras model.predict 返回的是 softmax 后的概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_preds_softmax.append(probabilities.cpu().numpy())

    # 将所有批次的预测结果合并
    test_Y_hat = np.concatenate(all_preds_softmax, axis=0)

    # 使用你已有的 mltools 进行评估
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')

    mltools.plot_confusion_matrix(cm, labels=classes, save_filename='figure/pytorch_lstm_total_confusion.png')
    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)


print("\n--- Starting Evaluation ---")
# 需要重新创建一个模型实例来加载权重，或者直接用训练结束后的模型
predict_pytorch(model, test_loader, device, filepath)

print("Process finished.")