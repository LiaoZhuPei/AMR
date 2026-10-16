"""
train_mamba.py
===================
使用 Mamba 模型训练调制识别任务。
基于原始 CNN-BiLSTM-Attention 训练脚本修改，数据处理流程完全不变。

改动说明：
  1. from mamba_model import build_model   （换模型文件）
  2. 权重保存路径改为 mamba_multirate_3.pt
  3. 混合精度：官方 mamba-ssm 支持 AMP；纯 PyTorch 实现同样支持
  4. 其余超参、DataLoader、评估逻辑均不变
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools

# ★ 唯一改动：引用 Mamba 模型文件
from mamba_model import build_model

# ============================================================
# AMP 兼容层（PyTorch 新旧 API 统一）
# ============================================================
try:
    from torch.amp import autocast, GradScaler
    def make_autocast():
        return autocast(device_type='cuda')
    scaler_class = lambda: GradScaler(device='cuda')
except TypeError:
    # 旧版 PyTorch 回退
    from torch.cuda.amp import autocast, GradScaler
    def make_autocast():
        return autocast()
    scaler_class = GradScaler

# ============================================================
# 1. 环境设置
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True
print(f"Using device: {device}")

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'Pi4DQPSK',
                '16QAM', '2ASK', '16APSK', '32APSK',
                '64QAM', '32QAM']
# ============================================================
# 2. 功率归一化（与原始完全一致）
# ============================================================
def power_normalize(signal_data):
    """
    功率归一化：将信号平均功率归一化为 1
    输入 shape: (2, 1024)
    """
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


# ============================================================
# 3. 数据集（与原始完全一致）
# ============================================================
class IQDataset(Dataset):
    def __init__(self, iq_data, labels, normalize=True, augment=False):
        self.raw_data = iq_data
        self.labels   = torch.from_numpy(labels).long()
        self.normalize = normalize
        self.augment   = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, sample):
        # 1. 随机时移
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            sample = np.roll(sample, shift, axis=1)

        # 2. 添加高斯白噪声
        if np.random.random() < 0.5:
            signal_power = np.mean(sample ** 2)
            snr_db = np.random.uniform(10, 30)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), sample.shape)
            sample = sample + noise

        # 3. 随机相位旋转
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
            sample = power_normalize(sample)

        sample_tensor = torch.from_numpy(sample).float()
        sample_tensor = sample_tensor.permute(1, 0)   # (2, 1024) → (1024, 2)

        return sample_tensor, self.labels[idx]


# ============================================================
# 4. 加载数据（与原始完全一致）
# ============================================================
print("Loading data...")
data = np.load('/sionna_dataset/Sionna1024_1.npz', allow_pickle=True)

train_raw        = data['X_train']                         # (N, 2, 1024)
train_labels_idx = data['Y_train']                         # (N,)
test_raw         = data['X_test']                          # (M, 2, 1024)
test_labels_idx  = data['Y_test']                          # (M,)
Z_test           = data['snr_test'].reshape(-1, 1)         # (M, 1)
Y_test_onehot    = np.eye(len(classes), dtype='uint8')[test_labels_idx]

print(f"Train raw: {train_raw.shape}, Test raw: {test_raw.shape}")

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx,
    test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_raw)}")

# ============================================================
# 5. 创建 DataLoader（与原始完全一致）
# ============================================================
batch_size  = 256
NUM_WORKERS = 0   # Windows 用 0；Linux 可设 4~8

train_dataset = IQDataset(X_train, Y_train, normalize=True, augment=True)
val_dataset   = IQDataset(X_val,   Y_val,   normalize=True, augment=False)
test_dataset  = IQDataset(test_raw, test_labels_idx, normalize=True, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ============================================================
# 6. 模型 / 损失 / 优化器 / 调度器
# ============================================================
# ★ 构建 Mamba 模型
#   - use_official=True  → 优先使用官方 mamba-ssm（速度更快）
#   - use_official=False → 强制使用纯 PyTorch 实现（无需额外安装）
model = build_model(
    classes=len(classes),
    d_model=128,        # Mamba 隐藏维度
    n_layers=4,         # Mamba Block 数量
    d_state=16,         # SSM 状态维度
    d_conv=4,           # 因果卷积核大小
    expand=2,           # 内部扩展比
    dropout=0.3,
    use_official=True   # 若 mamba-ssm 未安装，自动回退到纯 PyTorch
).to(device)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

nb_epoch = 200

# OneCycleLR
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.003,
    epochs=nb_epoch,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=100
)

# AMP 混合精度
scaler = scaler_class()

# ============================================================
# 7. 训练循环（与原始完全一致）
# ============================================================
filepath = 'weights/mamba_sionna1024_1.pt'
os.makedirs('weights', exist_ok=True)
os.makedirs('figure',  exist_ok=True)

best_val_loss          = float('inf')
patience_counter       = 0
early_stopping_patience = 30
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(nb_epoch):
    # ---------- 训练 ----------
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Train]")

    for inputs, labels in train_pbar:
        inputs = inputs.to(device, non_blocking=True)   # (batch, 1024, 2)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with make_autocast():
            outputs = model(inputs)                      # (batch, num_classes)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()   # OneCycleLR: 每个 batch 调用

        train_loss   += loss.item() * inputs.size(0)
        _, predicted  = torch.max(outputs.data, 1)
        train_total  += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        train_pbar.set_postfix({
            'loss': f'{train_loss / train_total:.4f}',
            'acc':  f'{train_correct / train_total:.4f}',
            'lr':   f'{current_lr:.6f}'
        })

    avg_train_loss = train_loss / train_total
    avg_train_acc  = train_correct / train_total

    # ---------- 验证 ----------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Val]")

    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with make_autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            val_loss    += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_pbar.set_postfix({
                'val_loss': f'{val_loss / val_total:.4f}',
                'val_acc':  f'{val_correct / val_total:.4f}'
            })

    avg_val_loss = val_loss / val_total
    avg_val_acc  = val_correct / val_total

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    print(f"  → Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    if avg_val_loss < best_val_loss:
        print(f"  ✓ Val loss improved ({best_val_loss:.4f} → {avg_val_loss:.4f}). Saving model...")
        torch.save(model.state_dict(), filepath)
        best_val_loss  = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ✗ No improvement. EarlyStopping: {patience_counter}/{early_stopping_patience}")
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

print("Training finished.")
mltools.show_history(history['train_loss'], history['val_loss'],
                     history['train_acc'],  history['val_acc'])


# ============================================================
# 8. 测试评估（与原始完全一致）
# ============================================================
def evaluate(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            with make_autocast():
                outputs = model(inputs)
            probs = torch.softmax(outputs.float(), dim=1)
            all_preds.append(probs.cpu().numpy())

    test_Y_hat = np.concatenate(all_preds, axis=0)

    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'Overall Accuracy: {acc * 100:.2f}% ({right}/{right + wrong})')

    mltools.plot_confusion_matrix(
        cm, labels=classes,
        save_filename='figure/mamba_confusion.png'
    )
    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)


print("\n--- Starting Evaluation ---")
evaluate(model, test_loader, device, filepath)
print("Done.")
