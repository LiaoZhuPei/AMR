import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools  # 确保 mltools.py 在工作目录中

# 导入您的 LSTM 模型
import LSTM as lstm_model

# ===================================================================
# --- 1. 配置参数 (请根据您的实际情况修改) ---
# ===================================================================

# --- 模式选择 ---
# 'A': 冻结特征层，只训练分类头 (更安全，防止遗忘)
# 'B': 微调整个网络 (更灵活，效果可能更好)
FINETUNE_MODE = 'A'  # <--- 在这里选择 'A' 或 'B'

# --- 路径配置 ---
ORIGINAL_MODEL_PATH = 'weights/pytorch_lstm4_weights_iq_normalized_13claA.pt'  # <-- 您原始的11分类模型
NEW_MODEL_SAVE_PATH = f'weights/finetuned_lstm_mode_{FINETUNE_MODE}.pt'
if not os.path.exists('weights'):
    os.makedirs('weights')

# --- 新的、合并后的完整数据集路径 (包含所有13个类别) ---
COMBINED_TRAIN_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_all_13_classes/train/train.csv'
COMBINED_TRAIN_LABELS_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_all_13_classes/train/train_labels.csv'

# --- 评估用的完整测试集路径 (也包含所有13个类别) ---
TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_all_13_classes/test/test.csv'
TEST_LABELS_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_all_13_classes/test/test_labels.csv'
TEST_SNR_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataset_all_13_classes/test/test_snr.csv'

# --- 类别配置 ---
# 原始模型训练时使用的类别
ORIGINAL_CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                    '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK','64QAM','32QAM']
# 新的完整类别列表 (顺序必须与标签文件中的索引对应)
ALL_CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
               '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK', '64QAM','32QAM', 'AM', 'FM']  # <-- 确认顺序

# --- 训练超参数 ---
LEARNING_RATE = 1e-4  # 方案 A 的学习率
VERY_LOW_LEARNING_RATE = 1e-6  # 方案 B 的学习率
BATCH_SIZE = 400
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 30

# --- 设备配置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 复用您的代码 (无需修改) ---
def per_sample_normalize_single(signal_data):
    min_val, max_val = np.min(signal_data), np.max(signal_data)
    range_val = max_val - min_val
    return np.zeros_like(signal_data) if range_val == 0 else 2 * (signal_data - min_val) / range_val - 1


class IQDataset(Dataset):
    def __init__(self, iq_data, labels):
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = per_sample_normalize_single(self.raw_data[idx])
        sample_tensor = torch.from_numpy(sample).float().permute(1, 0)
        return sample_tensor, self.labels[idx]


# ===================================================================
# --- 2. 核心训练与评估逻辑 ---
# ===================================================================
if __name__ == "__main__":

    # (A-1) 直接加载完整的训练数据
    print("--- Phase 1: Data Loading (from combined dataset) ---")
    IQ_CHANNELS, SIGNAL_LENGTH = 2, 1024

    try:
        train_data_flat = pd.read_csv(COMBINED_TRAIN_DATA_PATH, header=None).values
        train_raw = train_data_flat.reshape(-1, IQ_CHANNELS, SIGNAL_LENGTH)
        train_labels_idx = pd.read_csv(COMBINED_TRAIN_LABELS_PATH, header=None).values.flatten()
        print(f"Loaded {len(train_raw)} total training samples from the combined dataset.")
    except FileNotFoundError:
        print(f"Error: Training data not found at '{COMBINED_TRAIN_DATA_PATH}'. Please check the path.")
        exit()

    # (A-2) 划分训练集和验证集
    X_train, X_val, Y_train, Y_val = train_test_split(
        train_raw, train_labels_idx, test_size=0.2, random_state=42, stratify=train_labels_idx
    )
    train_dataset = IQDataset(X_train, Y_train)
    val_dataset = IQDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")

    # (A-3) 加载预训练模型并修改结构 (核心步骤)
    print("\n--- Phase 2: Model Preparation ---")
    # 步骤1: 实例化一个与【原始模型】结构完全相同的模型 (11个输出)
    model = lstm_model.build_model(classes=len(ORIGINAL_CLASSES))
    # 步骤2: 加载您训练好的【原始权重】
    model.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=device))

    # 步骤3: 替换最后一层以适应【新的类别总数】
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(ALL_CLASSES))
    print(f"Replaced 'classifier' layer. New output classes: {len(ALL_CLASSES)}")
    model = model.to(device)

    # (A-4) 选择策略并配置优化器
    criterion = nn.CrossEntropyLoss()

    if FINETUNE_MODE == 'A':
        print("\n--- Applying Strategy A: Freezing feature layers ---")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        print("Optimizer will only update the 'classifier' layer.")

    elif FINETUNE_MODE == 'B':
        print("\n--- Applying Strategy B: Fine-tuning all layers ---")
        optimizer = optim.Adam(model.parameters(), lr=VERY_LOW_LEARNING_RATE)
        print(f"Optimizer will update ALL layers with low learning rate: {VERY_LOW_LEARNING_RATE}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    # (A-5) 训练循环 (逻辑不变)
    print("\n--- Phase 3: Fine-tuning ---")
    # ... (这部分代码和上一个版本完全相同，无需修改，直接复用)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Finetune Train]")
        for inputs, labels in train_pbar:
            inputs, labels, optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted.cpu() == labels).sum().item()
            train_pbar.set_postfix(
                {'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted.cpu() == labels).sum().item()

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        print(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc * 100:.2f}%")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            print(f"Val loss improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            torch.save(model.state_dict(), NEW_MODEL_SAVE_PATH)
            best_val_loss, patience_counter = avg_val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nFine-tuning finished. Best model saved to '{NEW_MODEL_SAVE_PATH}'")

    # (A-6) 评估部分 (逻辑不变，但数据源已是完整测试集)
    print("\n--- Phase 4: Final Evaluation on Test Set ---")
    # ... (这部分代码和上一个版本完全相同，无需修改，直接复用)
    print("Loading comprehensive test data...")
    try:
        test_data_flat = pd.read_csv(TEST_DATA_PATH, header=None).values
        test_raw = test_data_flat.reshape(-1, IQ_CHANNELS, SIGNAL_LENGTH)
        test_labels_idx = pd.read_csv(TEST_LABELS_PATH, header=None).values.flatten()
        Y_test_onehot = np.eye(len(ALL_CLASSES), dtype='uint8')[test_labels_idx]
        Z_test = pd.read_csv(TEST_SNR_PATH, header=None).values

        test_dataset = IQDataset(test_raw, test_labels_idx)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"Loaded {len(test_raw)} test samples.")
    except FileNotFoundError:
        print("Test data files not found. Skipping final evaluation.")
        exit()

    eval_model = lstm_model.build_model(classes=len(ALL_CLASSES))
    eval_model.load_state_dict(torch.load(NEW_MODEL_SAVE_PATH, map_location=device))
    eval_model.to(device)
    eval_model.eval()

    all_preds_softmax = []
    predict_pbar = tqdm(test_loader, desc="Predicting")
    with torch.no_grad():
        for inputs, _ in predict_pbar:
            outputs = eval_model(inputs.to(device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_preds_softmax.append(probabilities.cpu().numpy())
    test_Y_hat = np.concatenate(all_preds_softmax, axis=0)

    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, ALL_CLASSES)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy on Test Set: {acc * 100:.2f}%')

    conf_matrix_filename = f'figure/finetuned_lstm_mode_{FINETUNE_MODE}_confusion.png'
    if not os.path.exists('figure'):
        os.makedirs('figure')
    mltools.plot_confusion_matrix(cm, labels=ALL_CLASSES, save_filename=conf_matrix_filename)

    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, ALL_CLASSES, min_snr=-18)

    print("\nProcess finished.")