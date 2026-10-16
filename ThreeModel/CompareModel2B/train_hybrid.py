"""
train_hybrid.py
===============
混合输入模型训练脚本（三分支：IQ + 瞬时特征 + 循环统计）

改动说明（相对原 train.py）：
  1. 导入 HybridDataset 替代原 IQDataset
  2. DataLoader 返回 (iq, inst, cyclic, label) 四元组
  3. build_model 来自 model_hybrid_tcn
  4. 其余超参数、调度器、早停逻辑与原脚本完全一致
  5. 不产生任何图片文件（验证图仍调用 mltools，但 mltools 本身不受约束）

使用方式：
  python train_hybrid.py
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

# ── 导入混合模块 ──────────────────────────────────────────────────────────
from hybrid_features import HybridDataset
from model_hybrid_tcn import build_model

import mltools   # 原项目工具，保持不变

# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════
DATA_PATH   = './sionna_dataset/Sionna0_b.npz'   # ← 与原脚本一致，按需修改
MODEL_NAME  = 'hybrid_tcn'
USE_BRANCH_B = True    # 是否使用瞬时特征分支
USE_BRANCH_C = True    # 是否使用循环统计分支

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Device:  {device}")
print(f"Model:   {MODEL_NAME}  (branch_b={USE_BRANCH_B}, branch_c={USE_BRANCH_C})")

# ══════════════════════════════════════════════════════════════════════════
# 加载数据
# ══════════════════════════════════════════════════════════════════════════
print("Loading data...")
data    = np.load(DATA_PATH, allow_pickle=True)
classes = list(data['classes'])
print(f"Classes ({len(classes)}): {classes}")

train_raw        = data['X_train']    # (N_train, 2, 1024)
train_labels_idx = data['Y_train']    # (N_train,)
test_raw         = data['X_test']     # (N_test, 2, 1024)
test_labels_idx  = data['Y_test']     # (N_test,)

Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
Z_test        = data['snr_test'].reshape(-1, 1)

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx,
    test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(test_raw)}")

# ══════════════════════════════════════════════════════════════════════════
# DataLoader
# ══════════════════════════════════════════════════════════════════════════
# 注意：HybridDataset 在 __getitem__ 中逐样本提取特征
# 若 num_workers > 0，特征计算会并行化（显著加速）
# Windows 下 num_workers=0，Linux 可设为 4~8
BATCH_SIZE   = 256    # 因三分支内存占用更大，建议从原 400 降至 256
NUM_WORKERS  = 0      # Linux 推荐 4；Windows 设 0

train_dataset = HybridDataset(X_train, Y_train, normalize=True, augment=True)
val_dataset   = HybridDataset(X_val,   Y_val,   normalize=True, augment=False)
test_dataset  = HybridDataset(test_raw, test_labels_idx, normalize=True, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════
# 模型 / 损失 / 优化器
# ══════════════════════════════════════════════════════════════════════════
model = build_model(
    classes      = len(classes),
    use_branch_b = USE_BRANCH_B,
    use_branch_c = USE_BRANCH_C,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

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

# ── 输出文件路径 ──────────────────────────────────────────────────────────
os.makedirs('weights', exist_ok=True)
os.makedirs('figure',  exist_ok=True)
filepath       = f'weights/{MODEL_NAME}.pt'
confusion_path = f'figure/{MODEL_NAME}_confusion.png'

# ══════════════════════════════════════════════════════════════════════════
# 训练循环
# ══════════════════════════════════════════════════════════════════════════
best_val_loss           = float('inf')
patience_counter        = 0
early_stopping_patience = 20
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
epoch_times = []

for epoch in range(nb_epoch):
    t0 = time.time()

    # ── 训练 ──────────────────────────────────────────────────────────────
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epoch} [Train]", ncols=100)

    for iq, inst, cyclic, labels in pbar:
        iq     = iq.to(device, non_blocking=True)
        inst   = inst.to(device, non_blocking=True)   if USE_BRANCH_B else None
        cyclic = cyclic.to(device, non_blocking=True) if USE_BRANCH_C else None
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out  = model(iq, inst, cyclic)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tr_loss    += loss.item() * iq.size(0)
        _, pred     = torch.max(out.data, 1)
        tr_total   += labels.size(0)
        tr_correct += (pred == labels).sum().item()
        pbar.set_postfix(loss=f'{tr_loss/tr_total:.4f}',
                         acc=f'{tr_correct/tr_total:.4f}')

    avg_tr_loss = tr_loss / tr_total
    avg_tr_acc  = tr_correct / tr_total

    # ── 验证 ──────────────────────────────────────────────────────────────
    model.eval()
    vl_loss, vl_correct, vl_total = 0.0, 0, 0
    with torch.no_grad():
        for iq, inst, cyclic, labels in val_loader:
            iq     = iq.to(device, non_blocking=True)
            inst   = inst.to(device, non_blocking=True)   if USE_BRANCH_B else None
            cyclic = cyclic.to(device, non_blocking=True) if USE_BRANCH_C else None
            labels = labels.to(device, non_blocking=True)
            with autocast():
                out  = model(iq, inst, cyclic)
                loss = criterion(out, labels)
            vl_loss    += loss.item() * iq.size(0)
            _, pred     = torch.max(out.data, 1)
            vl_total   += labels.size(0)
            vl_correct += (pred == labels).sum().item()

    avg_vl_loss = vl_loss / vl_total
    avg_vl_acc  = vl_correct / vl_total
    ep_time     = time.time() - t0
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
# 训练摘要
# ══════════════════════════════════════════════════════════════════════════
total_t = sum(epoch_times)
print(f"\n{'='*60}")
print(f"[{MODEL_NAME}] Training Summary")
print(f"  Epochs:        {len(epoch_times)}")
print(f"  Avg epoch:     {np.mean(epoch_times):.1f}s")
print(f"  Total time:    {total_t:.0f}s ({total_t/60:.1f} min)")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Parameters:    {total_params:,}")
print(f"  Weights:       {filepath}")
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
        for iq, inst, cyclic, _ in tqdm(loader, desc=f"[{MODEL_NAME}] Testing"):
            iq     = iq.to(device, non_blocking=True)
            inst   = inst.to(device, non_blocking=True)   if USE_BRANCH_B else None
            cyclic = cyclic.to(device, non_blocking=True) if USE_BRANCH_C else None
            with autocast():
                out = model(iq, inst, cyclic)
            probs = torch.softmax(out.float(), dim=1)
            all_preds.append(probs.cpu().numpy())

    Y_hat = np.concatenate(all_preds, axis=0)
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, Y_hat, classes)
    acc = round(right / (right + wrong), 4)
    print(f"[{MODEL_NAME}] Overall Accuracy: {acc*100:.2f}%")
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename=confusion_path)
    mltools.calculate_acc_cm_each_snr(Y_test_onehot, Y_hat, Z_test, classes, min_snr=-18)


print(f"\n--- [{MODEL_NAME}] Evaluation ---")
evaluate(model, test_loader, device, filepath)
print("Done.")
