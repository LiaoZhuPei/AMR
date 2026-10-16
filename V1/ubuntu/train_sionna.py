"""
统一训练脚本：只需修改 MODEL_NAME 即可切换模型
所有其他设置（数据、优化器、调度器、增强）完全相同
数据源：generate_sionna_data_pro.py 生成的 .npz 文件
"""
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

# ╔══════════════════════════════════════════════════════════════╗
# ║  唯一需要修改的地方：选择模型                                   ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_NAME = 'transformer'  # 选项: 'transformer', 'tcn', 'mamba'

# ============================================================
# 根据 MODEL_NAME 自动导入对应模型
# ============================================================
if MODEL_NAME == 'transformer':
    from model_transformer import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn import build_model
elif MODEL_NAME == 'mamba':
    from model_mamba import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}. Choose from 'transformer', 'tcn', 'mamba'")

# ============================================================
# 环境设置（所有模型相同）
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
print(f"Using device: {device}")
print(f"Selected model: {MODEL_NAME}")

# ============================================================
# 数据预处理（所有模型相同）
# ============================================================
def power_normalize(signal_data):
    """功率归一化"""
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


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
            sample = power_normalize(sample)
        sample_tensor = torch.from_numpy(sample).float()
        sample_tensor = sample_tensor.permute(1, 0)  # (2,1024) → (1024,2)
        return sample_tensor, self.labels[idx]


# ============================================================
# 加载数据（所有模型相同）
# ============================================================
print("Loading data...")

DATA_PATH = './sionna_dataset/SionnaAll_1.npz'
data = np.load(DATA_PATH, allow_pickle=True)

# 从 npz 中读取类别列表
classes = list(data['classes'])
print(f"Classes ({len(classes)}):{classes}")

# load train
train_raw = data['X_train']  # (N_train, 2, 1024)
train_labels_idx = data['Y_train']  # (N_train,)

# load test
test_raw = data['X_test']  # (N_test, 2, 1024)
test_labels_idx = data['Y_test']  # (N_test,)

# 测试集 one-hot 和 SNR
Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
Z_test = data['snr_test'].reshape(-1, 1)

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx, test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test:{len(test_raw)}")


# ============================================================
# DataLoader（所有模型相同）
# ============================================================
batch_size = 256
NUM_WORKERS = 8

train_dataset = IQDataset(X_train, Y_train, normalize=True, augment=True)
val_dataset = IQDataset(X_val, Y_val, normalize=True, augment=False)
test_dataset = IQDataset(test_raw, test_labels_idx, normalize=True, augment=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# ============================================================
# 模型 / 损失 / 优化器 / 调度器（所有模型相同的超参数）
# ============================================================
model = build_model(classes=len(classes)).to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[{MODEL_NAME}] Trainable parameters: {total_params:,}")

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

# ============================================================
# 自动生成文件名（根据 MODEL_NAME）
# ============================================================
filepath = f'weights/{MODEL_NAME}_sionnaAll_1.pt'
confusion_path = f'figure/{MODEL_NAME}_confusion.png'
os.makedirs('weights', exist_ok=True)
os.makedirs('figure', exist_ok=True)

# ============================================================
# 训练循环（所有模型完全相同）
# ============================================================
best_val_loss = float('inf')
patience_counter = 0
early_stopping_patience = 30
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

epoch_times = []  # 记录每个 epoch 的时间，用于速度对比

for epoch in range(nb_epoch):
    epoch_start = time.time()

    # ---------- 训练 ----------
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epoch} [Train]")

    for inputs, labels in train_pbar:
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
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        train_pbar.set_postfix({
            'loss': f'{train_loss / train_total:.4f}',
            'acc': f'{train_correct / train_total:.4f}',
        })

    avg_train_loss = train_loss / train_total
    avg_train_acc = train_correct / train_total

    # ---------- 验证 ----------
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
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / val_total
    avg_val_acc = val_correct / val_total

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    print(f"  [{MODEL_NAME}] Epoch {epoch + 1}: "
          f"Train={avg_train_acc:.4f} Val={avg_val_acc:.4f} "
          f"Time={epoch_time:.1f}s LR={optimizer.param_groups[0]['lr']:.6f}")

    if avg_val_loss < best_val_loss:
        print(f"  ✓ Saved. ({best_val_loss:.4f} → {avg_val_loss:.4f})")
        torch.save(model.state_dict(), filepath)
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping.")
            break
 

# ============================================================
# 训练总结
# ============================================================
avg_epoch_time = np.mean(epoch_times)
total_time = np.sum(epoch_times)
print(f"\n{'=' * 60}")
print(f"[{MODEL_NAME}] Training Summary")
print(f"  Total epochs:      {len(epoch_times)}")
print(f"  Avg epoch time:    {avg_epoch_time:.1f}s")
print(f"  Total time:        {total_time:.0f}s ({total_time / 60:.1f} min)")
print(f"  Best val loss:     {best_val_loss:.4f}")
print(f"  Parameters:        {total_params:,}")
print(f"  Weights saved to:  {filepath}")
print(f"{'=' * 60}")

mltools.show_history(history['train_loss'], history['val_loss'],
                     history['train_acc'], history['val_acc'])

# ============================================================
# 测试评估
# ============================================================
def evaluate(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc=f"[{MODEL_NAME}] Testing"):
            inputs = inputs.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
            probs = torch.softmax(outputs.float(), dim=1)
            all_preds.append(probs.cpu().numpy())

    test_Y_hat = np.concatenate(all_preds, axis=0)
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'[{MODEL_NAME}] Overall Accuracy: {acc * 100:.2f}%')
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename=confusion_path)
    mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)


print(f"\n--- [{MODEL_NAME}] Evaluation ---")
evaluate(model, test_loader, device, filepath)
print("Done.")