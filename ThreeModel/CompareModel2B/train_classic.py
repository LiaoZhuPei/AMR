"""
train_classic.py
----------------
使用经典 CV 骨干网络（ResNet50 / EfficientNet-B3 / MobileNetV3-Large / DenseNet121）
训练 IQ 调制信号分类模型。

修改方式：只需改动 § CONFIG 中的两个变量
  MODEL_NAME  : 选择模型
  USE_PRETRAINED : 是否使用 ImageNet 预训练权重
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

# 本地模块
from classic_models import build_classic_model

# ============================================================
# § CONFIG —— 只需修改这里
# ============================================================
MODEL_NAME     = 'resnet50'          # 可选: resnet50 | efficientnet_b3 | mobilenet_v3_large | densenet121
USE_PRETRAINED = True                # True=迁移学习(ImageNet权重)  False=从零训练
DATA_PATH      = './sionna_dataset/SionnaA_6.npz'
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True
print(f"Using device : {device}")
print(f"Model        : {MODEL_NAME}  (pretrained={USE_PRETRAINED})")


# ============================================================
# 1. 功率归一化
# ============================================================
def power_normalize(signal_data: np.ndarray) -> np.ndarray:
    """
    功率归一化：将信号平均功率归一化为 1
    输入 shape: (2, 1024)
    """
    I, Q  = signal_data[0], signal_data[1]
    power = np.mean(I**2 + Q**2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


# ============================================================
# 2. 数据集（含数据增强）
# ============================================================
class IQDataset(Dataset):
    def __init__(self, iq_data, labels, normalize=True, augment=False):
        self.raw_data = iq_data
        self.labels   = torch.from_numpy(labels).long()
        self.normalize = normalize
        self.augment   = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, sample: np.ndarray) -> np.ndarray:
        # 随机时移
        if np.random.random() < 0.5:
            shift  = np.random.randint(-50, 50)
            sample = np.roll(sample, shift, axis=1)

        # 加高斯白噪声
        if np.random.random() < 0.5:
            sig_pwr   = np.mean(sample**2)
            snr_db    = np.random.uniform(10, 30)
            noise_pwr = sig_pwr / (10**(snr_db / 10))
            noise     = np.random.normal(0, np.sqrt(noise_pwr), sample.shape)
            sample    = sample + noise

        # 随机相位旋转
        if np.random.random() < 0.5:
            theta     = np.random.uniform(0, 2 * np.pi)
            I, Q      = sample[0].copy(), sample[1].copy()
            sample[0] = I * np.cos(theta) - Q * np.sin(theta)
            sample[1] = I * np.sin(theta) + Q * np.cos(theta)

        return sample

    def __getitem__(self, idx):
        sample = self.raw_data[idx].copy()

        if self.augment:
            sample = self._augment(sample)
        if self.normalize:
            sample = power_normalize(sample)

        # (2, 1024) → (1024, 2)  与 classic_models 中适配层的期望一致
        tensor = torch.from_numpy(sample).float().permute(1, 0)
        return tensor, self.labels[idx]


# ============================================================
# 3. 加载数据
# ============================================================
print("\nLoading data...")
data    = np.load(DATA_PATH, allow_pickle=True)
classes = list(data['classes'])
print(f"Classes ({len(classes)}): {classes}")

train_raw        = data['X_train']   # (N_train, 2, 1024)
train_labels_idx = data['Y_train']   # (N_train,)
test_raw         = data['X_test']    # (N_test,  2, 1024)
test_labels_idx  = data['Y_test']    # (N_test,)

Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
Z_test        = data['snr_test'].reshape(-1, 1)

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx,
    test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_raw)}")


# ============================================================
# 4. DataLoader
# ============================================================
BATCH_SIZE  = 256          # 经典模型参数量更大，适当降低 batch size
NUM_WORKERS = 0            # Windows: 0；Linux: 4~8

train_dataset = IQDataset(X_train, Y_train, normalize=True, augment=True)
val_dataset   = IQDataset(X_val,   Y_val,   normalize=True, augment=False)
test_dataset  = IQDataset(test_raw, test_labels_idx, normalize=True, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ============================================================
# 5. 构建模型 / 损失 / 优化器 / 调度器
# ============================================================
model = build_classic_model(
    model_name  = MODEL_NAME,
    num_classes = len(classes),
    pretrained  = USE_PRETRAINED
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 迁移学习策略：适配层+分类头用大 lr，骨干其余层用小 lr
if USE_PRETRAINED:
    # 先收集 adapter 和分类头的参数 id（用 id() 去重，避免同一参数进两个组）
    fast_ids = set()
    fast_params, slow_params = [], []

    for name, param in model.named_parameters():
        # adapter 层 或 骨干最后的分类头（backbone.fc / backbone.classifier）
        is_fast = name.startswith('adapter.') or name.startswith('backbone.fc') \
                  or name.startswith('backbone.classifier')
        if is_fast:
            fast_ids.add(id(param))
            fast_params.append(param)
        else:
            slow_params.append(param)

    optimizer = optim.AdamW([
        {'params': fast_params, 'lr': 3e-4},   # 适配层 + 分类头
        {'params': slow_params, 'lr': 1e-4},   # 骨干其余层
    ], weight_decay=1e-4)
    print("[Optimizer] Differential LR: adapter+head=3e-4, backbone=1e-4")
    NUM_GROUPS = 2
else:
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print("[Optimizer] Uniform LR: 1e-3")
    NUM_GROUPS = 1

NB_EPOCH  = 500
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr         = [3e-4, 1e-4] if USE_PRETRAINED else [3e-3],
    epochs         = NB_EPOCH,
    steps_per_epoch= len(train_loader),
    pct_start      = 0.1,
    anneal_strategy= 'cos',
    div_factor     = 10,
    final_div_factor= 100
)

scaler = GradScaler()


# ============================================================
# 6. 训练循环
# ============================================================
WEIGHT_DIR  = 'weights'
FIGURE_DIR  = 'figure'
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# 权重文件名包含模型名和是否预训练，方便多次实验
pretrain_tag = 'pretrained' if USE_PRETRAINED else 'scratch'
SAVE_PATH    = os.path.join(WEIGHT_DIR, f'{MODEL_NAME}_{pretrain_tag}_SionnaA_6.pt')

best_val_loss         = float('inf')
patience_counter      = 0
EARLY_STOP_PATIENCE   = 25
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print(f"\nSave path : {SAVE_PATH}")
print(f"Epochs    : {NB_EPOCH}  |  Early stop patience: {EARLY_STOP_PATIENCE}\n")

for epoch in range(NB_EPOCH):
    # ---------- Train ----------
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NB_EPOCH} [Train]")

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss    += loss.item() * inputs.size(0)
        _, predicted   = torch.max(outputs.data, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f'{train_loss/train_total:.4f}',
            'acc' : f'{train_correct/train_total:.4f}',
            'lr'  : f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    avg_train_loss = train_loss / train_total
    avg_train_acc  = train_correct / train_total

    # ---------- Validation ----------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    pbar_v = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NB_EPOCH} [Val  ]")

    with torch.no_grad():
        for inputs, labels in pbar_v:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            val_loss    += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            pbar_v.set_postfix({
                'val_loss': f'{val_loss/val_total:.4f}',
                'val_acc' : f'{val_correct/val_total:.4f}'
            })

    avg_val_loss = val_loss / val_total
    avg_val_acc  = val_correct / val_total

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    print(f"  → Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    if avg_val_loss < best_val_loss:
        print(f"  ✓ Val loss improved ({best_val_loss:.4f} → {avg_val_loss:.4f}). Saving...")
        torch.save(model.state_dict(), SAVE_PATH)
        best_val_loss    = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ✗ No improvement. EarlyStopping: {patience_counter}/{EARLY_STOP_PATIENCE}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

print("\nTraining finished.")

# 可选：如果你有 mltools，取消注释
import mltools
mltools.show_history(history['train_loss'], history['val_loss'],
                     history['train_acc'],  history['val_acc'])


# ============================================================
# 7. 测试评估
# ============================================================
def evaluate(model, test_loader, device, model_path):
    print(f"\nLoading best weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Testing"):
            inputs  = inputs.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
            probs = torch.softmax(outputs.float(), dim=1)
            all_preds.append(probs.cpu().numpy())

    test_Y_hat = np.concatenate(all_preds, axis=0)

    # 若无 mltools，改用简单精度统计
    try:
        import mltools
        cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
        acc = round(right / (right + wrong), 4)
        print(f'Overall Accuracy: {acc*100:.2f}%  ({right}/{right+wrong})')
        mltools.plot_confusion_matrix(
            cm, labels=classes,
            save_filename=f'figure/{MODEL_NAME}_{pretrain_tag}_confusion.png'
        )
        mltools.calculate_acc_cm_each_snr(Y_test_onehot, test_Y_hat, Z_test, classes, min_snr=-18)
    except ImportError:
        # mltools 不存在时的简化版
        preds  = np.argmax(test_Y_hat, axis=1)
        truths = np.argmax(Y_test_onehot, axis=1)
        acc    = np.mean(preds == truths)
        print(f'Overall Accuracy (simple): {acc*100:.2f}%')


print("\n--- Starting Evaluation ---")
evaluate(model, test_loader, device, SAVE_PATH)
print("Done.")
