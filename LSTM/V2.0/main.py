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
import CNNLSTM as mcl

# --- 设备设置 (不变) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 数据加载和预处理 (这是主要修改区域) ---

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4QPSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK','64QAM','32QAM']

# 定义数据参数，这对于从CSV重塑回原始形状至关重要
IQ_CHANNELS = 2
SIGNAL_LENGTH = 1024

# 训练数据
print("Loading training data...")
train_data_flat = pd.read_csv(
    'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train.csv',
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
    'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test.csv',
    header=None
).values

# 重塑测试数据
num_test_samples = test_data_flat.shape[0]
test_data = test_data_flat.reshape(num_test_samples, IQ_CHANNELS, SIGNAL_LENGTH)

# PyTorch需要 [N, C, H, W] 格式, 当前为 [N, H, W]
# 我们需要增加一个通道维度 C=1
train_data = np.expand_dims(train_data, axis=1)  # Shape: [N, 1, 2, 1024]
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
train_labels_df = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train_labels.csv',
                              header=None)
train_labels_idx = train_labels_df.values.flatten()

test_labels_df = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_labels.csv', header=None)
test_labels_idx = test_labels_df.values.flatten()


# 为了和mltools中的函数兼容，我们仍然需要one-hot版本用于评估
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


Y_test_onehot = to_categorical(test_labels_idx, len(classes))

# SNR
train_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train_snr.csv', header=None).values
test_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_snr.csv', header=None).values
Z_test = test_snr

# 划分训练集和验证集
X_train, X_val, Y_train_idx, Y_val_idx = train_test_split(
    train_data, train_labels_idx, test_size=0.2, random_state=42
)
print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

# --- 2. 创建PyTorch DataLoader ---
batch_size = 400

# 将Numpy数组转换为Tensor
X_train_t = torch.from_numpy(X_train).float()
Y_train_t = torch.from_numpy(Y_train_idx).long()
X_val_t = torch.from_numpy(X_val).float()
Y_val_t = torch.from_numpy(Y_val_idx).long()
X_test_t = torch.from_numpy(test_data).float()

# 创建Dataset和DataLoader
train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset = TensorDataset(X_val_t, Y_val_t)
# test loader的label部分用onehot，方便mltools中的函数直接使用
test_dataset = TensorDataset(X_test_t, torch.from_numpy(Y_test_onehot).float())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# --- 3. 模型、损失函数、优化器定义 ---
nb_epoch = 300
model = mcl.build_model(classes=len(classes)).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器 (替代ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

# --- 4. 训练循环 ---
filepath = 'weights/pytorch_lstmCNN_2410_normalized_DataCla12_A.pt.pt'  # PyTorch模型权重文件
if not os.path.exists('weights'):
    os.makedirs('weights')

# 用于早停和模型保存的变量
best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 20

# 用于记录历史指标
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(nb_epoch):
    # ** 训练 **
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    # 使用tqdm包装train_loader来显示进度条
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

        # 在进度条上实时更新指标
        train_pbar.set_postfix({
            'loss': f'{train_loss / train_total:.4f}',
            'acc': f'{train_correct / train_total:.4f}'
        })

    avg_train_loss = train_loss / train_total
    avg_train_acc = train_correct / train_total

    # ** 验证 **
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    # 使用tqdm包装val_loader
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

            # 在进度条上实时更新验证指标
            val_pbar.set_postfix({
                'val_loss': f'{val_loss / val_total:.4f}',
                'val_acc': f'{val_correct / val_total:.4f}'
            })

    avg_val_loss = val_loss / val_total
    avg_val_acc = val_correct / val_total

    # 记录历史
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    # 学习率调度
    scheduler.step(avg_val_loss)

    # ** 模型保存和早停 **
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

# 训练结束后绘制和保存历史曲线
print("Training finished. Plotting history...")
mltools.show_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])


# --- 5. 评估函数 ---
def predict(model, test_loader, device):
    # 加载最佳权重
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()

    all_preds = []
    # 使用tqdm包装test_loader
    predict_pbar = tqdm(test_loader, desc="Predicting on Test Set")
    with torch.no_grad():
        for inputs, _ in predict_pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 应用softmax得到概率
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_preds.append(probabilities.cpu().numpy())

    # 将所有批次的预测结果拼接起来
    test_Y_hat = np.concatenate(all_preds, axis=0)

    # 使用mltools进行评估
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right} correct / {wrong + right} total)')

    mltools.plot_confusion_matrix(cm, labels=classes, save_filename='figure/pytorch_cnn_total_confusion.png')

    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)


# 执行评估
print("\n--- Starting Evaluation ---")
predict(model, test_loader, device)

print("Process finished.")