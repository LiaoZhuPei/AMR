"""
train_classic.py  (v2)
----------------------
使用专为 1D 时序信号设计的经典模型训练 IQ 调制分类器。

§ CONFIG 只需改两个变量：
  MODEL_NAME : 选择模型
  DATA_PATH  : 数据集路径
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from classic_models_1d import build_classic_model

# ============================================================
# § CONFIG
# ============================================================
MODEL_NAME = 'densenet1d'   # resnet1d | inception1d | densenet1d | tcn
DATA_PATH  = './sionna_dataset/Sionna0_b.npz'
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Device : {device}  |  Model : {MODEL_NAME}")


# ============================================================
# 功率归一化（与测试脚本保持完全一致）
# ============================================================
def power_normalize(signal_data: np.ndarray) -> np.ndarray:
    I, Q  = signal_data[0], signal_data[1]
    power = np.mean(I**2 + Q**2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


# ============================================================
# 数据集
# ============================================================
class IQDataset(Dataset):
    def __init__(self, iq_data, labels, augment=False):
        self.raw_data = iq_data
        self.labels   = torch.from_numpy(labels).long()
        self.augment  = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, sample):
        if np.random.random() < 0.5:
            sample = np.roll(sample, np.random.randint(-50, 50), axis=1)
        if np.random.random() < 0.5:
            snr_db    = np.random.uniform(10, 30)
            noise_pwr = np.mean(sample**2) / (10**(snr_db/10))
            sample    = sample + np.random.normal(0, np.sqrt(noise_pwr), sample.shape)
        if np.random.random() < 0.5:
            theta     = np.random.uniform(0, 2*np.pi)
            I, Q      = sample[0].copy(), sample[1].copy()
            sample[0] = I*np.cos(theta) - Q*np.sin(theta)
            sample[1] = I*np.sin(theta) + Q*np.cos(theta)
        return sample

    def __getitem__(self, idx):
        sample = self.raw_data[idx].copy()
        if self.augment:
            sample = self._augment(sample)
        sample = power_normalize(sample)
        # (2,1024) → (1024,2)
        tensor = torch.from_numpy(sample).float().permute(1, 0)
        return tensor, self.labels[idx]


# ============================================================
# 加载数据
# ============================================================
print("Loading data...")
data    = np.load(DATA_PATH, allow_pickle=True)
classes = list(data['classes'])
print(f"Classes ({len(classes)}): {classes}")

train_raw        = data['X_train']
train_labels_idx = data['Y_train']
test_raw         = data['X_test']
test_labels_idx  = data['Y_test']
Y_test_onehot    = np.eye(len(classes), dtype='uint8')[test_labels_idx]
Z_test           = data['snr_test'].reshape(-1, 1)

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx,
    test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_raw)}")


# ============================================================
# DataLoader
# ============================================================
BATCH_SIZE  = 512    # 1D 轻量模型显存占用小，可用更大 batch
NUM_WORKERS = 0      # Windows: 0；Linux: 4~8

train_loader = DataLoader(IQDataset(X_train, Y_train, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(IQDataset(X_val,   Y_val,   augment=False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(IQDataset(test_raw, test_labels_idx, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ============================================================
# 模型 / 损失 / 优化器 / 调度器
# ============================================================
model     = build_classic_model(MODEL_NAME, num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

NB_EPOCH  = 500
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr          = 3e-3,
    epochs          = NB_EPOCH,
    steps_per_epoch = len(train_loader),
    pct_start       = 0.1,
    anneal_strategy = 'cos',
    div_factor      = 10,
    final_div_factor= 100
)
scaler = GradScaler()


# ============================================================
# 训练循环
# ============================================================
os.makedirs('weights', exist_ok=True)
os.makedirs('figure',  exist_ok=True)
SAVE_PATH = f'weights/{MODEL_NAME}_Sionna0_b.pt'

best_val_loss       = float('inf')
patience_counter    = 0
EARLY_STOP_PATIENCE = 25
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print(f"\nSave path : {SAVE_PATH}")
print(f"Batch size: {BATCH_SIZE}  |  Epochs: {NB_EPOCH}  |  Early stop: {EARLY_STOP_PATIENCE}\n")

for epoch in range(NB_EPOCH):
    # ---------- Train ----------
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>3}/{NB_EPOCH} [Train]", leave=False)

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tr_loss    += loss.item() * inputs.size(0)
        _, pred     = torch.max(outputs.data, 1)
        tr_total   += labels.size(0)
        tr_correct += (pred == labels).sum().item()
        pbar.set_postfix(loss=f'{tr_loss/tr_total:.4f}',
                         acc=f'{tr_correct/tr_total:.4f}',
                         lr=f'{optimizer.param_groups[0]["lr"]:.5f}')

    avg_tr_loss = tr_loss / tr_total
    avg_tr_acc  = tr_correct / tr_total

    # ---------- Validation ----------
    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            va_loss    += loss.item() * inputs.size(0)
            _, pred     = torch.max(outputs.data, 1)
            va_total   += labels.size(0)
            va_correct += (pred == labels).sum().item()

    avg_va_loss = va_loss / va_total
    avg_va_acc  = va_correct / va_total

    history['train_loss'].append(avg_tr_loss)
    history['train_acc'].append(avg_tr_acc)
    history['val_loss'].append(avg_va_loss)
    history['val_acc'].append(avg_va_acc)

    print(f"Epoch {epoch+1:>3}/{NB_EPOCH}  "
          f"tr_loss={avg_tr_loss:.4f}  tr_acc={avg_tr_acc:.4f}  "
          f"va_loss={avg_va_loss:.4f}  va_acc={avg_va_acc:.4f}  "
          f"lr={optimizer.param_groups[0]['lr']:.5f}")

    if avg_va_loss < best_val_loss:
        print(f"  ✓ Saved ({best_val_loss:.4f} → {avg_va_loss:.4f})")
        torch.save(model.state_dict(), SAVE_PATH)
        best_val_loss    = avg_va_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping.")
            break

print("Training finished.")

# 可选：mltools 可视化
#import mltools
#mltools.show_history(history['train_loss'], history['val_loss'],
#                     history['train_acc'],  history['val_acc'])


# ============================================================
# 测试评估
# ============================================================
def evaluate(model, loader, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            with autocast():
                out = model(inputs)
            all_preds.append(torch.softmax(out.float(), dim=1).cpu().numpy())
    Y_hat = np.concatenate(all_preds, axis=0)

    try:
        import mltools
        cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, Y_hat, classes)
        acc = right / (right + wrong)
        print(f'Accuracy: {acc*100:.2f}%  ({right}/{right+wrong})')
        mltools.plot_confusion_matrix(cm, labels=classes,
                                      save_filename=f'figure/{MODEL_NAME}_confusion.png')
        mltools.calculate_acc_cm_each_snr(Y_test_onehot, Y_hat, Z_test, classes, min_snr=-18)
    except ImportError:
        preds  = np.argmax(Y_hat, axis=1)
        truths = np.argmax(Y_test_onehot, axis=1)
        print(f'Accuracy: {np.mean(preds==truths)*100:.2f}%')

print("\n--- Evaluation ---")
evaluate(model, test_loader, SAVE_PATH)
print("Done.")
