import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools
import LSTM as lstm

# --- 设备设置 (不变) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 数据加载和预处理 ---

classes = ['16QAM', '32QAM', '64QAM']

# 定义数据参数
IQ_CHANNELS = 2
SIGNAL_LENGTH = 1024

# 训练数据
print("Loading training data...")
train_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam6/train/train.csv',
    header=None
).values
num_train_samples = train_data_flat.shape[0]
train_data = train_data_flat.reshape(num_train_samples, IQ_CHANNELS, SIGNAL_LENGTH)

# 测试数据
print("Loading test data from CSV...")
test_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam6/test/test.csv',
    header=None
).values
num_test_samples = test_data_flat.shape[0]
test_data = test_data_flat.reshape(num_test_samples, IQ_CHANNELS, SIGNAL_LENGTH)

# PyTorch需要 [N, C, H, W] 格式, 当前为 [N, H, W]
# 我们需要增加一个通道维度 C=1
train_data = np.expand_dims(train_data, axis=1)  # Shape: [N, 1, 2, 1024]
test_data = np.expand_dims(test_data, axis=1)  # Shape: [N, 1, 2, 1024]
print("Data shapes adjusted for PyTorch.")


##### 修改点 1: 定义逐样本归一化函数 #####
def per_sample_normalize(data_array):
    """
    对每个样本（独立的时间序列）进行归一化，将其缩放到 [-1, 1] 区间。
    输入 shape: (N, C, H, W) -> (N, 1, 2, 1024)
    """
    print("Performing per-sample normalization...")
    # 计算每个样本在 (C, H, W) 维度上的 min 和 max
    # keepdims=True 使得 min/max 的 shape 为 (N, 1, 1, 1)，便于广播
    min_vals = np.min(data_array, axis=(1, 2, 3), keepdims=True)
    max_vals = np.max(data_array, axis=(1, 2, 3), keepdims=True)

    # 计算范围
    range_vals = max_vals - min_vals

    # 处理范围为0的情况（例如，信号是常数）
    # np.where(condition, x, y) 如果条件为真，返回x，否则返回y
    # 这里我们避免除以零
    normalized_data = 2 * (data_array - min_vals) / np.where(range_vals == 0, 1, range_vals) - 1

    # 对于范围为0的样本，上述操作会得到 -1。可以将其设置为0。
    # data_array - min_vals 在这种情况下已经是0了。
    normalized_data[range_vals.squeeze() == 0] = 0

    print("Per-sample normalization complete.")
    return normalized_data


##### 修改点 2: 应用逐样本归一化，并移除旧的标准化代码 #####
# 移除旧的标准化代码
# train_mean = np.mean(train_data)
# train_std = np.std(train_data)
# train_data = (train_data - train_mean) / train_std
# test_data = (test_data - train_mean) / train_std

# 应用新的逐样本归一化
train_data = per_sample_normalize(train_data)
test_data = per_sample_normalize(test_data)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++


# 标签 (不变)
train_labels_df = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam6/train/train_labels.csv',
                              header=None)
train_labels_idx = train_labels_df.values.flatten()
test_labels_df = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam6/test/test_labels.csv', header=None)
test_labels_idx = test_labels_df.values.flatten()


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


Y_test_onehot = to_categorical(test_labels_idx, len(classes))

# SNR (不变)
train_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam6/train/train_snr.csv', header=None).values
test_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_qam6/test/test_snr.csv', header=None).values
Z_test = test_snr

# 划分训练集和验证集 (不变)
X_train, X_val, Y_train_idx, Y_val_idx = train_test_split(
    train_data, train_labels_idx, test_size=0.2, random_state=42
)
print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

# --- 2. 创建PyTorch DataLoader (不变) ---
batch_size = 500

X_train_t = torch.from_numpy(X_train).float()
Y_train_t = torch.from_numpy(Y_train_idx).long()
X_val_t = torch.from_numpy(X_val).float()
Y_val_t = torch.from_numpy(Y_val_idx).long()
X_test_t = torch.from_numpy(test_data).float()

train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset = TensorDataset(X_val_t, Y_val_t)
test_dataset = TensorDataset(X_test_t, torch.from_numpy(Y_test_onehot).float())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# --- 3. 模型、损失函数、优化器定义 (不变) ---
nb_epoch = 300
model = lstm.build_model(classes=len(classes)).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

# --- 4. 训练循环 (不变) ---
filepath = 'weights/weights_lstm_qam_norm.pt'  # 建议为新方法使用新文件名
if not os.path.exists('weights'):
    os.makedirs('weights')

best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 50
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
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_pbar.set_postfix({
            'loss': f'{train_loss / train_total:.4f}',
            'acc': f'{train_correct / train_total:.4f}'
        })
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
            val_pbar.set_postfix({
                'val_loss': f'{val_loss / val_total:.4f}',
                'val_acc': f'{val_correct / val_total:.4f}'
            })
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
        if patience_counter >= early_stopping_patience:
            print(f"EarlyStopping counter: {patience_counter}/{early_stopping_patience}. Early stopping triggered.")
            break
        else:
            print(f"EarlyStopping counter: {patience_counter}/{early_stopping_patience}")

print("Training finished. Plotting history...")
mltools.show_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])


# --- 5. 评估函数 (不变, 但确保它加载正确的权重) ---
def predict(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds = []
    predict_pbar = tqdm(test_loader, desc="Predicting on Test Set")
    with torch.no_grad():
        for inputs, _ in predict_pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_preds.append(probabilities.cpu().numpy())
    test_Y_hat = np.concatenate(all_preds, axis=0)

    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')

    mltools.plot_confusion_matrix(cm, labels=classes, save_filename='figure/pytorch_lstm_persample_confusion.png')
    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)


# 执行评估
print("\n--- Starting Evaluation ---")
predict(model, test_loader, device, filepath)  # 确保传入正确的filepath

print("Process finished.")