"""
train_cnn.py  (v2 — 抗过拟合版)
=================================
针对严重过拟合（train≈0.9, val≈0.7）的修改：

  1. Mixup 数据增强：在 batch 级别对特征张量做线性插值，
     强迫模型学习平滑决策边界而非记忆样本
  2. 增强幅度加大：时移范围扩大，新增幅度随机缩放
  3. weight_decay 从 1e-4 提高到 3e-4
  4. label_smoothing 从 0.1 提高到 0.15
  5. 早停耐心从 30 降到 20，避免在过拟合区域浪费时间
  6. 新增每 epoch 打印 train-val gap，便于监控

其余（数据路径、DataLoader、调度器结构）与原脚本一致。
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
DATA_PATH   = 'E:/Project/AMR-Benchmark/HisarMod/Pytorch_Sequence/ModelCompare2B/sionna_dataset/Sionna0_a.npz'
MODEL_NAME  = 'simple_cnn'
REGULARIZE  = 'medium'    # 'light' | 'medium' | 'heavy'
USE_EAGER   = True
BATCH_SIZE  = 512
NUM_WORKERS = 4           # Windows 请设 0

# ── 正则化超参（集中在此处，方便调整）────────────────────────────────────
MIXUP_ALPHA      = 0.2    # Mixup 强度；0.0 = 关闭
WEIGHT_DECAY     = 3e-4   # 原为 1e-4
LABEL_SMOOTHING  = 0.15   # 原为 0.1
EARLY_STOP_PAT   = 20     # 原为 30

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Device:     {device}")
print(f"Model:      {MODEL_NAME}  regularize={REGULARIZE}")
print(f"Mixup α:    {MIXUP_ALPHA}")
print(f"WeightDecay:{WEIGHT_DECAY}  LabelSmooth:{LABEL_SMOOTHING}")

# ══════════════════════════════════════════════════════════════════════════
# Mixup 工具
# ══════════════════════════════════════════════════════════════════════════

def mixup_batch(feats: torch.Tensor, labels: torch.Tensor,
                alpha: float, num_classes: int):
    """
    对一个 batch 做 Mixup。
    返回: (mixed_feats, label_a, label_b, lam)
    损失计算: lam * ce(out, label_a) + (1-lam) * ce(out, label_b)
    """
    if alpha <= 0.0:
        return feats, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    B   = feats.size(0)
    idx = torch.randperm(B, device=feats.device)
    mixed = lam * feats + (1.0 - lam) * feats[idx]
    return mixed, labels, labels[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════════════════
# 加载数据
# ══════════════════════════════════════════════════════════════════════════
print("\nLoading raw IQ data...")
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
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(test_raw)}")

# ══════════════════════════════════════════════════════════════════════════
# 特征提取 + DataLoader
# ══════════════════════════════════════════════════════════════════════════
print("\nExtracting features...")
t0 = time.time()

if USE_EAGER:
    train_dataset = EagerFeatureDataset(X_train, Y_train, normalize=True)
    val_dataset   = EagerFeatureDataset(X_val,   Y_val,   normalize=True)
    test_dataset  = EagerFeatureDataset(test_raw, test_labels_idx, normalize=True)
    nw = 0
else:
    train_dataset = FeatureDataset(X_train, Y_train, normalize=True, augment=True)
    val_dataset   = FeatureDataset(X_val,   Y_val,   normalize=True, augment=False)
    test_dataset  = FeatureDataset(test_raw, test_labels_idx, normalize=True, augment=False)
    nw = NUM_WORKERS

print(f"  Done in {time.time()-t0:.1f}s")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=nw, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=nw, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=nw, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════
# 模型 / 优化器 / 调度器
# ══════════════════════════════════════════════════════════════════════════
model = build_model(classes=len(classes), regularize=REGULARIZE).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=0.001,
                        weight_decay=WEIGHT_DECAY)

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
filepath       = f'weights/{MODEL_NAME}_Sionna0_a_2.pt'
confusion_path = f'figure/{MODEL_NAME}_confusion_Sionna0_a_2.png'

# ══════════════════════════════════════════════════════════════════════════
# 训练循环
# ══════════════════════════════════════════════════════════════════════════
best_val_loss    = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
epoch_times      = []

for epoch in range(nb_epoch):
    t_ep = time.time()

    # ── train ─────────────────────────────────────────────────────────────
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    pbar = tqdm(train_loader,
                desc=f"Epoch {epoch+1}/{nb_epoch} [Train]", ncols=100)

    for feats, labels in pbar:
        feats  = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Mixup ────────────────────────────────────────────────────────
        mixed, y_a, y_b, lam = mixup_batch(
            feats, labels, MIXUP_ALPHA, len(classes))

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out  = model(mixed)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tr_loss  += loss.item() * feats.size(0)
        # 精度用原始标签（非 mixed）计算，方便监控
        with torch.no_grad():
            _, pred = torch.max(model(feats), 1)
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
    gap         = avg_tr_acc - avg_vl_acc      # ← 过拟合监控
    ep_time     = time.time() - t_ep
    epoch_times.append(ep_time)

    history['train_loss'].append(avg_tr_loss)
    history['train_acc'].append(avg_tr_acc)
    history['val_loss'].append(avg_vl_loss)
    history['val_acc'].append(avg_vl_acc)

    # gap 超过 0.15 时给出警告
    gap_flag = '  ⚠ gap>{:.2f}'.format(gap) if gap > 0.15 else ''
    print(f"  Epoch {epoch+1:3d}: "
          f"train={avg_tr_acc:.4f}  val={avg_vl_acc:.4f}  "
          f"gap={gap:+.4f}{gap_flag}  "
          f"t={ep_time:.1f}s  lr={optimizer.param_groups[0]['lr']:.2e}")

    if avg_vl_loss < best_val_loss:
        print(f"  ✓ Saved ({best_val_loss:.4f} → {avg_vl_loss:.4f})")
        torch.save(model.state_dict(), filepath)
        best_val_loss = avg_vl_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PAT:
            print(f"Early stopping (patience={EARLY_STOP_PAT}).")
            break

# ══════════════════════════════════════════════════════════════════════════
# 摘要
# ══════════════════════════════════════════════════════════════════════════
total_t = sum(epoch_times)
final_gap = history['train_acc'][-1] - history['val_acc'][-1]
print(f"\n{'='*60}")
print(f"[{MODEL_NAME}] Summary")
print(f"  Epochs:        {len(epoch_times)}")
print(f"  Best val loss: {best_val_loss:.4f}")
print(f"  Final gap:     {final_gap:+.4f}  "
      f"({'良好' if final_gap < 0.05 else '轻微过拟合' if final_gap < 0.10 else '仍过拟合，考虑 heavy 模式'})")
print(f"  Parameters:    {total_params:,}")
print(f"  Total time:    {total_t:.0f}s ({total_t/60:.1f} min)")
print(f"{'='*60}\n")

mltools.show_history(history['train_loss'], history['val_loss'],
                     history['train_acc'],  history['val_acc'])

# ══════════════════════════════════════════════════════════════════════════
# 评估
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
