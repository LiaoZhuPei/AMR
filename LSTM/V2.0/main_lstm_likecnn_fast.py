import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import mltools  # 假设 mltools 是你自己的工具库
import LSTM as lstm_model

# --- 1. 全局定义: 设备、类别、函数和类 ---
# 这些部分保持在全局作用域，以便子进程可以导入它们。

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4QPSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK', '64QAM', '32QAM']


def per_sample_normalize_single(signal_data):
    """
    对单个样本（独立的时间序列）进行归一化，将其缩放到 [-1, 1] 区间。
    输入 shape: (2, 1024)
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data)
    normalized_data = 2 * (signal_data - min_val) / range_val - 1
    return normalized_data


class IQDataset(Dataset):
    def __init__(self, iq_data, labels, apply_normalization=True):
        """
        Args:
            iq_data (numpy.ndarray): 原始 I/Q 数据, shape (N, 2, 1024).
            labels (numpy.ndarray): 标签数据.
            apply_normalization (bool): 是否应用逐样本归一化.
        """
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.apply_normalization = apply_normalization
        print("Dataset created.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        if self.apply_normalization:
            sample = per_sample_normalize_single(sample)
        sample_tensor = torch.from_numpy(sample).float()
        sample_tensor = sample_tensor.permute(1, 0)
        return sample_tensor, self.labels[idx]


# --- ★★★ 关键修改：将所有执行逻辑放入 main 块 ★★★ ---
# 这是在Windows上使用多进程数据加载（num_workers > 0）所必需的。
if __name__ == '__main__':
    print(f"Using device: {device}")

    # --- 4. 数据加载 ---
    print("Loading data from .csv files...")
    IQ_CHANNELS = 2
    SIGNAL_LENGTH = 1024

    train_data_flat = pd.read_csv(
        'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train.csv',
        header=None
    ).values
    num_train_samples = train_data_flat.shape[0]
    train_raw = train_data_flat.reshape(num_train_samples, IQ_CHANNELS, SIGNAL_LENGTH)
    print(f"Loaded {num_train_samples} training samples. Shape: {train_raw.shape}")

    test_data_flat = pd.read_csv(
        'E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test.csv',
        header=None
    ).values
    num_test_samples = test_data_flat.shape[0]
    test_raw = test_data_flat.reshape(num_test_samples, IQ_CHANNELS, SIGNAL_LENGTH)
    print(f"Loaded {num_test_samples} test samples. Shape: {test_raw.shape}")

    train_labels_idx = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train_labels.csv',
                                   header=None).values.flatten()
    test_labels_idx = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_labels.csv',
                                  header=None).values.flatten()
    Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]

    train_snr = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/train/train_snr.csv',
                            header=None).values
    Z_test = pd.read_csv('E:/Project/AMR-Benchmark/HisarMod/dataset/data12claE/test/test_snr.csv', header=None).values

    # 划分训练集和验证集
    X_train, X_val, Y_train, Y_val = train_test_split(
        train_raw, train_labels_idx, test_size=0.2, random_state=42, stratify=train_labels_idx
    )
    print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

    # 创建 Dataset 和 DataLoader
    batch_size = 400
    num_workers = 4  # 设置为大于0的数以启用多进程加载

    train_dataset = IQDataset(X_train, Y_train, apply_normalization=True)
    val_dataset = IQDataset(X_val, Y_val, apply_normalization=True)
    test_dataset = IQDataset(test_raw, test_labels_idx, apply_normalization=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    # --- 5. 模型、损失函数、优化器定义 ---
    model = lstm_model.build_model(classes=len(classes)).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    # --- 6. 训练循环 ---
    nb_epoch = 500
    filepath = 'weights/pytorch_lstm_2410_normalized_DataCla12_fast_A.pt'
    if not os.path.exists('weights'):
        os.makedirs('weights')

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 初始化 GradScaler 用于自动混合精度训练
    scaler = GradScaler()

    for epoch in range(nb_epoch):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            optimizer.zero_grad(set_to_none=True)

            # 使用 autocast 上下文管理器进行前向传播
            with autocast():
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            # 使用 scaler 进行反向传播
            scaler.scale(loss).backward()

            # 在 optimizer.step() 之前 unscale 梯度
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新权重
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_pbar.set_postfix(
                {'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # 验证时也可以使用 autocast 以获得微小的性能提升
                with autocast():
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
        mltools.plot_confusion_matrix(cm, labels=classes,
                                      save_filename='figure/pytorch_lstm_iq_norm_total_confusion.png')
        mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)


    print("\n--- Starting Evaluation ---")
    predict_pytorch(model, test_loader, device, filepath)

    print("Process finished.")