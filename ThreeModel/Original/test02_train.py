import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools
from test01_model import build_model  # 你之前的 CNN+BiLSTM+Attention 模型

# ============================================================
# 环境
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print(f"Using device: {device}")

classes = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']


# ============================================================
# 归一化：改回 [-1, 1] 的 min-max 归一化
# ============================================================
def per_sample_normalize_single(signal_data):
    """
    对单个样本进行 [-1, 1] 归一化
    输入 shape: (2, 1024)
    
    与你原始代码中 per_sample_normalize 的逻辑完全一致
    只是从批处理版本改为单样本版本（在 Dataset 的 __getitem__ 中调用）
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val

    if range_val == 0:
        return np.zeros_like(signal_data, dtype=np.float32)

    normalized_data = 2 * (signal_data - min_val) / range_val - 1
    return normalized_data.astype(np.float32)


# ============================================================
# 数据集
# ============================================================
class IQDataset(Dataset):
    def __init__(self, iq_data, labels, normalize=True, augment=False):
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.normalize = normalize
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, sample):
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            sample = np.roll(sample, shift, axis=1)
        if np.random.random() < 0.5:
            signal_power = np.mean(sample ** 2)
            if signal_power > 1e-10:
                snr_db = np.random.uniform(10, 30)
                noise_power = signal_power / (10 ** (snr_db / 10))
                sample = sample + np.random.normal(0, np.sqrt(noise_power), sample.shape)
        if np.random.random() < 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            I, Q = sample[0].copy(), sample[1].copy()
            sample[0] = I * np.cos(theta) - Q * np.sin(theta)
            sample[1] = I * np.sin(theta) + Q * np.cos(theta)
        return sample

    def __getitem__(self, idx):
        sample = self.raw_data[idx].copy()
        if self.augment:
            sample = self._augment(sample)
        if self.normalize:
            sample = per_sample_normalize_single(sample)  # ← [-1,1] 归一化
        sample_tensor = torch.from_numpy(sample).float()
        sample_tensor = sample_tensor.permute(1, 0)  # (2,1024) → (1024,2)
        return sample_tensor, self.labels[idx]


# ============================================================
# 加载数据
# ============================================================
print("Loading data...")
DATA_ROOT = 'E:/Project/AMR-Benchmark/HisarMod/dataset/dataMatA1'

train_raw = pd.read_csv(f'{DATA_ROOT}/train/train.csv', header=None).values.reshape(-1, 2, 1024)
test_raw = pd.read_csv(f'{DATA_ROOT}/test/test.csv', header=None).values.reshape(-1, 2, 1024)
train_labels_idx = pd.read_csv(f'{DATA_ROOT}/train/train_labels.csv', header=None).values.flatten()
test_labels_idx = pd.read_csv(f'{DATA_ROOT}/test/test_labels.csv', header=None).values.flatten()
Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
Z_test = pd.read_csv(f'{DATA_ROOT}/test/test_snr.csv', header=None).values

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx, test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_raw)}")

# ============================================================
# DataLoader
# ============================================================
batch_size = 400
train_loader = DataLoader(
    IQDataset(X_train, Y_train, normalize=True, augment=True),
    batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(
    IQDataset(X_val, Y_val, normalize=True, augment=False),
    batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(
    IQDataset(test_raw, test_labels_idx, normalize=True, augment=False),
    batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ============================================================
# 模型 / 损失 / 优化器
# ============================================================
model = build_model(classes=len(classes)).to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

nb_epoch = 200
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.003, epochs=nb_epoch,
    steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos',
    div_factor=10, final_div_factor=100
)
scaler = GradScaler()

filepath = 'weights/cnn_bilstm_attn_minmax.pt'  # 区分归一化方式
os.makedirs('weights', exist_ok=True)
os.makedirs('figure', exist_ok=True)

# ============================================================
# 训练
# ============================================================
best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 30
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
epoch_times = []

for epoch in range(nb_epoch):
    epoch_start = time.time()

    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epoch} [Train]")

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (pred == labels).sum().item()
        pbar.set_postfix(loss=f'{train_loss/train_total:.4f}',
                         acc=f'{train_correct/train_total:.4f}')

    avg_train_loss = train_loss / train_total
    avg_train_acc = train_correct / train_total

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()

    avg_val_loss = val_loss / val_total
    avg_val_acc = val_correct / val_total
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    print(f"  Train={avg_train_acc:.4f} Val={avg_val_acc:.4f} "
          f"Time={epoch_time:.1f}s LR={optimizer.param_groups[0]['lr']:.6f}")

    if avg_val_loss < best_val_loss:
        print(f"  ✓ Saved ({best_val_loss:.4f} → {avg_val_loss:.4f})")
        torch.save(model.state_dict(), filepath)
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping.")
            break

print(f"\nAvg epoch: {np.mean(epoch_times):.1f}s, "
      f"Total: {np.sum(epoch_times)/60:.1f}min")
mltools.show_history(history['train_loss'], history['val_loss'],
                     history['train_acc'], history['val_acc'])

# ============================================================
# 测试
# ============================================================
model.load_state_dict(torch.load(filepath, map_location=device))
model.eval()
all_preds = []
with torch.no_grad():
    for inputs, _ in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(device, non_blocking=True)
        with autocast():
            outputs = model(inputs)
        all_preds.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())

test_Y_hat = np.concatenate(all_preds, axis=0)
cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
print(f'Accuracy: {right/(right+wrong)*100:.2f}%')
mltools.plot_confusion_matrix(cm, labels=classes,
                              save_filename='figure/cnn_bilstm_attn_minmax_confusion.png')
mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)
print("Done.")