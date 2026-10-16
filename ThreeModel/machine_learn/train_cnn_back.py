"""
train_cnn.py
============
特征提取 + 简单 CNN 训练脚本

流程：
  1. 加载 .npz 原始 IQ 数据（与原 train.py 相同路径约定）
  2. 用 EagerFeatureDataset 一次性提取 8 通道特征并缓存（更快）
     —— 若内存不足，将 USE_EAGER 改为 False 切换懒加载
  3. 训练 SimpleCNN，超参与原脚本保持一致
  4. 评估并输出混淆矩阵
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from feature_extract import EagerFeatureDataset, FeatureDataset
from model_cnn import build_model

import mltools

# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════
DATA_PATH  = 'E:/Project/AMR-Benchmark/HisarMod/Pytorch_Sequence/ModelCompare2B/sionna_dataset/Sionna0_b.npz'   # ← 按需修改
MODEL_NAME = 'simple_cnn'

# True  = 构造时一次性提取全部特征（快，需要更多内存）
# False = __getitem__ 逐样本提取（省内存，稍慢）
USE_EAGER   = True
BATCH_SIZE  = 512      # 特征提取后张量更小，batch 可以更大
NUM_WORKERS = 4        # Windows 请设 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Device:     {device}")
print(f"Model:      {MODEL_NAME}")
print(f"Eager mode: {USE_EAGER}")

# ══════════════════════════════════════════════════════════════════════════
# 加载原始数据
# ══════════════════════════════════════════════════════════════════════════
print("\nLoading raw IQ data...")
data    = np.load(DATA_PATH, allow_pickle=True)
classes = list(data['classes'])
print(f"Classes ({len(classes)}): {classes}")

train_raw        = data['X_train']     # (N_train, 2, 1024)
train_labels_idx = data['Y_train']
test_raw         = data['X_test']      # (N_test,  2, 1024)
test_labels_idx  = data['Y_test']
Y_test_onehot    = np.eye(len(classes), dtype='uint8')[test_labels_idx]
Z_test           = data['snr_test'].reshape(-1, 1)

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx,
    test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(test_raw)}")

# ══════════════════════════════════════════════════════════════════════════
# 特征提取 + DataLoader
# ══════════════════════════════════════════════════════════════════════════
print("\nExtracting features...")
t0 = time.time()

if USE_EAGER:
    # 一次性提取，训练时 DataLoader 无额外 CPU 开销
    train_dataset = EagerFeatureDataset(X_train, Y_train, normalize=True)
    val_dataset   = EagerFeatureDataset(X_val,   Y_val,   normalize=True)
    test_dataset  = EagerFeatureDataset(test_raw, test_labels_idx, normalize=True)
    # Eager 模式下已缓存，不需要多进程再提特征
    nw_train, nw_val = 0, 0
else:
    # 懒加载（训练集开启增强）
    train_dataset = FeatureDataset(X_train, Y_train, normalize=True, augment=True)
    val_dataset   = FeatureDataset(X_val,   Y_val,   normalize=True, augment=False)
    test_dataset  = FeatureDataset(test_raw, test_labels_idx, normalize=True, augment=False)
    nw_train, nw_val = NUM_WORKERS, NUM_WORKERS

print(f"  Done in {time.time()-t0:.1f}s")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=nw_train, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=nw_val,   pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=nw_val,   pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════
# 模型 / 优化器 / 调度器
# ══════════════════════════════════════════════════════════════════════════
model = build_model(classes=len(classes)).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

nb_epoch  = 200
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.003, epochs=nb_epoch,
    steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos',
    div_factor=10, final_div_factor=100
)
scaler = GradScaler()

os.makedirs('weights', exist_ok=True)
os.makedirs('figure',  exist_ok=True)
filepath       = f'weights/{MODEL_NAME}_Sionna0_b.pt'
confusion_path = f'figure/{MODEL_NAME}_confusion_Sionna0_b.png'

# ══════════════════════════════════════════════════════════════════════════
# 训练循环
# ══════════════════════════════════════════════════════════════════════════
best_val_loss           = float('inf')
patience_counter        = 0
early_stopping_patience = 20
history     = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
epoch_times = []

for epoch in range(nb_epoch):
    t_ep = time.time()

    # ── train ─────────────────────────────────────────────────────────────
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epoch} [Train]", ncols=100)
    for feats, labels in pbar:
        feats  = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out  = model(feats)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        tr_loss    += loss.item() * feats.size(0)
        _, pred     = torch.max(out.data, 1)
        tr_total   += labels.size(0)
        tr_correct += (pred == labels).sum().item()
        pbar.set_postfix(loss=f'{tr_loss/tr_total:.4f}',
                         acc=f'{tr_correct/tr_total:.4f}')

    avg_tr_loss = tr_loss / tr_total
    avg_tr_acc  = tr_correct / tr_total

    # ── val ───────────────────────────────────────────────────────────────
    model.eval()
    vl_loss, vl_correct, vl_total = 0.0, 0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats  = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                out  = model(feats)
                loss = criterion(out, labels)
            vl_loss    += loss.item() * feats.size(0)
            _, pred     = torch.max(out.data, 1)
            vl_total   += labels.size(0)
            vl_correct += (pred == labels).sum().item()

    avg_vl_loss = vl_loss / vl_total
    avg_vl_acc  = vl_correct / vl_total
    ep_time     = time.time() - t_ep
    epoch_times.append(ep_time)

    history['train_loss'].append(avg_tr_loss)
    history['train_acc'].append(avg_tr_acc)
    history['val_loss'].append(avg_vl_loss)
    history['val_acc'].append(avg_vl_acc)

    print(f"  [{MODEL_NAME}] Epoch {epoch+1}: "
          f"Train={avg_tr_acc:.4f}  Val={avg_vl_acc:.4f}  "
          f"Time={ep_time:.1f}s  LR={optimizer.param_groups[0]['lr']:.6f}")

    if avg_vl_loss < best_val_loss:
        print(f"  ✓ Saved ({best_val_loss:.4f} → {avg_vl_loss:.4f})")
        torch.save(model.state_dict(), filepath)
        best_val_loss = avg_vl_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping.")
            break

# ══════════════════════════════════════════════════════════════════════════
# 摘要
# ══════════════════════════════════════════════════════════════════════════
total_t = sum(epoch_times)
print(f"\n{'='*60}")
print(f"[{MODEL_NAME}] Summary")
print(f"  Epochs:        {len(epoch_times)}")
print(f"  Avg epoch:     {np.mean(epoch_times):.1f}s")
print(f"  Total time:    {total_t:.0f}s ({total_t/60:.1f} min)")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Parameters:    {total_params:,}")
print(f"{'='*60}\n")

mltools.show_history(history['train_loss'], history['val_loss'],
                     history['train_acc'],  history['val_acc'])

# ══════════════════════════════════════════════════════════════════════════
# 测试评估
# ══════════════════════════════════════════════════════════════════════════
def evaluate(model, loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for feats, _ in tqdm(loader, desc=f"[{MODEL_NAME}] Testing"):
            feats = feats.to(device, non_blocking=True)
            with autocast():
                out = model(feats)
            probs = torch.softmax(out.float(), dim=1)
            all_preds.append(probs.cpu().numpy())
    Y_hat = np.concatenate(all_preds, axis=0)
    cm, right, wrong = mltools.calculate_confusion_matrix(
        Y_test_onehot, Y_hat, classes)
    acc = round(right / (right + wrong), 4)
    print(f"[{MODEL_NAME}] Overall Accuracy: {acc*100:.2f}%")
    mltools.plot_confusion_matrix(cm, labels=classes,
                                  save_filename=confusion_path)
    mltools.calculate_acc_cm_each_snr(
        Y_test_onehot, Y_hat, Z_test, classes, min_snr=-18)


print(f"\n--- [{MODEL_NAME}] Evaluation ---")
evaluate(model, test_loader, device, filepath)
print("Done.")
