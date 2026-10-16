"""
train_hierarchical_v2.py
两阶段层次识别训练脚本

损失函数设计:
  L_total = L_coarse + λ_fine * L_fine

  - L_coarse: 所有样本参与，Focal Loss，目标 = {0:非QAM, 1:QAM}
  - L_fine  : 仅 QAM 样本参与，Focal Loss，目标 = QAM 内部类别索引 (0-4)
  - λ_fine  : 细分类损失权重，默认 1.5（让模型更注重 QAM 细分）

训练结束后会分别报告:
  - 粗分类准确率（全集）
  - QAM 细分类准确率（仅 QAM 样本）
  - 每个 QAM 阶数的独立准确率
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mltools

from model_bilstm_v2_hierarchical import (
    build_model, ALL_CLASSES, QAM_CLASSES,
    make_coarse_label, make_fine_label
)

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
MODEL_NAME   = 'bilstm_hierarchical'
LAMBDA_FINE  = 1.5      # 细分类损失权重
MIXUP_PROB   = 0.5
MIXUP_ALPHA  = 0.2
BATCH_SIZE   = 400
NB_EPOCH     = 200
EARLY_STOP   = 30
NUM_WORKERS  = 0
DATA_PATH    = './sionna_dataset/SionnaA_6.npz'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Focal Loss（支持 ignore_index=-1 过滤非QAM）
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma        = gamma
        self.ls           = label_smoothing
        self.ignore_index = ignore_index
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        valid = (targets != self.ignore_index)
        if valid.sum() == 0:
            return logits.sum() * 0.0   # 全为 ignore，返回 0 梯度

        logits_v  = logits[valid]
        targets_v = targets[valid]
        weight    = self.weight if self.weight is not None else None

        ce    = F.cross_entropy(logits_v, targets_v,
                                weight=weight,
                                label_smoothing=self.ls, reduction='none')
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


# ─────────────────────────────────────────────
# 数据预处理
# ─────────────────────────────────────────────
def power_normalize(signal_data):
    I, Q   = signal_data[0], signal_data[1]
    power  = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


class HierarchicalIQDataset(Dataset):
    """
    每个样本同时提供三个标签:
      label_all    : 全类别索引 (0-13)
      label_coarse : 粗分类     (0=非QAM, 1=QAM)
      label_fine   : 细分类     (0-4=QAM内部, -1=非QAM)
    """
    def __init__(self, iq_data, labels_all, classes,
                 normalize=True, augment=False):
        self.raw_data     = iq_data
        self.classes      = classes
        self.normalize    = normalize
        self.augment      = augment

        n = len(labels_all)
        self.labels_all    = torch.from_numpy(labels_all).long()
        self.labels_coarse = torch.tensor(
            [make_coarse_label(y, classes) for y in labels_all], dtype=torch.long
        )
        self.labels_fine   = torch.tensor(
            [make_fine_label(y, classes) for y in labels_all], dtype=torch.long
        )

    def __len__(self):
        return len(self.labels_all)

    def _augment(self, sample):
        if np.random.random() < 0.5:
            shift  = np.random.randint(-25, 25)
            sample = np.roll(sample, shift, axis=1)
        if np.random.random() < 0.5:
            signal_power = np.mean(sample ** 2)
            if signal_power > 1e-10:
                snr_db     = np.random.uniform(10, 30)
                noise_power= signal_power / (10 ** (snr_db / 10))
                sample     = sample + np.random.normal(0, np.sqrt(noise_power), sample.shape)
        if np.random.random() < 0.5:
            theta     = np.random.uniform(0, 2 * np.pi)
            I, Q      = sample[0].copy(), sample[1].copy()
            sample[0] = I * np.cos(theta) - Q * np.sin(theta)
            sample[1] = I * np.sin(theta) + Q * np.cos(theta)
        if np.random.random() < 0.3:
            sample = sample * np.random.uniform(0.8, 1.2)
        return sample

    def __getitem__(self, idx):
        sample = self.raw_data[idx].copy()
        if self.augment:
            sample = self._augment(sample)
        if self.normalize:
            sample = power_normalize(sample)
        tensor = torch.from_numpy(sample).float().permute(1, 0)  # (L, 2)
        return tensor, self.labels_all[idx], self.labels_coarse[idx], self.labels_fine[idx]


# ─────────────────────────────────────────────
# Mixup（同时混合三组标签）
# ─────────────────────────────────────────────
def mixup_batch(x, yall, yc, yf, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], yall, yall[idx], yc, yc[idx], yf, yf[idx], lam


# ─────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────
print("Loading data...")
data = np.load(DATA_PATH, allow_pickle=True)
classes = list(data['classes'])
print(f"Classes ({len(classes)}): {classes}")

train_raw        = data['X_train']
train_labels_idx = data['Y_train']
test_raw         = data['X_test']
test_labels_idx  = data['Y_test']

Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
Z_test        = data['snr_test'].reshape(-1, 1)

X_train, X_val, Y_train, Y_val = train_test_split(
    train_raw, train_labels_idx,
    test_size=0.2, random_state=42, stratify=train_labels_idx
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_raw)}")

train_dataset = HierarchicalIQDataset(X_train, Y_train, classes, normalize=True, augment=True)
val_dataset   = HierarchicalIQDataset(X_val,   Y_val,   classes, normalize=True, augment=False)
test_dataset  = HierarchicalIQDataset(test_raw, test_labels_idx, classes, normalize=True, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ─────────────────────────────────────────────
# 模型 / 损失 / 优化器
# ─────────────────────────────────────────────
model = build_model(num_all_classes=len(classes), num_qam_classes=len(QAM_CLASSES)).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[{MODEL_NAME}] Trainable parameters: {total_params:,}")

# 粗分类损失 (2类: 非QAM=0 / QAM=1)
# QAM 样本较少，适当上调 QAM 类权重
coarse_weight = torch.tensor([1.0, 1.5], device=device)
criterion_coarse = FocalLoss(gamma=2.0, label_smoothing=0.05,
                             weight=coarse_weight, ignore_index=-1)

# 细分类损失 (5类，仅 QAM 样本, ignore_index=-1 过滤非QAM)
fine_weight = torch.ones(len(QAM_CLASSES), device=device)
# 高阶 QAM 适当加权
for i, name in enumerate(QAM_CLASSES):
    if name in ('128QAM', '256QAM', '64QAM'):
        fine_weight[i] = 1.5
criterion_fine = FocalLoss(gamma=2.0, label_smoothing=0.05,
                            weight=fine_weight, ignore_index=-1)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.003, epochs=NB_EPOCH,
    steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos',
    div_factor=10, final_div_factor=100
)
scaler = GradScaler()

filepath       = f'weights/{MODEL_NAME}_SionnaA_6_II.pt'
confusion_path = f'figure/{MODEL_NAME}_confusion.png'
os.makedirs('weights', exist_ok=True)
os.makedirs('figure',  exist_ok=True)

# ─────────────────────────────────────────────
# 训练循环
# ─────────────────────────────────────────────
best_val_acc     = 0.0
best_val_loss    = float('inf')
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
epoch_times = []


def compute_combined_loss(logits_c, logits_f, yc, yf, yc2=None, yf2=None, lam=1.0):
    """计算联合损失，支持 Mixup"""
    if yc2 is None:
        lc = criterion_coarse(logits_c, yc)
        lf = criterion_fine(logits_f, yf)
    else:
        lc = lam * criterion_coarse(logits_c, yc) + (1 - lam) * criterion_coarse(logits_c, yc2)
        lf = lam * criterion_fine(logits_f, yf)   + (1 - lam) * criterion_fine(logits_f, yf2)
    return lc + LAMBDA_FINE * lf


for epoch in range(NB_EPOCH):
    t0 = time.time()

    # ──── 训练 ────────────────────────────────
    model.train()
    train_loss = 0.0
    train_coarse_correct = train_coarse_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NB_EPOCH} [Train]")
    for inputs, yall, yc, yf in pbar:
        inputs = inputs.to(device, non_blocking=True)
        yc     = yc.to(device, non_blocking=True)
        yf     = yf.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if np.random.random() < MIXUP_PROB:
            inputs_m, _, _, yc_a, yc_b, yf_a, yf_b, lam = mixup_batch(
                inputs, yall.to(device), yc, yf, alpha=MIXUP_ALPHA
            )
            with autocast():
                lc_out, lf_out, _ = model(inputs_m)
                loss = compute_combined_loss(lc_out, lf_out, yc_a, yf_a, yc_b, yf_b, lam)
        else:
            with autocast():
                lc_out, lf_out, _ = model(inputs)
                loss = compute_combined_loss(lc_out, lf_out, yc, yf)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        current_scale = scaler.get_scale()

        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= current_scale:
            scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        with torch.no_grad():
            pred_c = lc_out.argmax(1)
            train_coarse_total   += yc.size(0)
            train_coarse_correct += (pred_c == yc).sum().item()

        pbar.set_postfix({
            'loss'     : f'{train_loss / train_coarse_total:.4f}',
            'coarse_acc': f'{train_coarse_correct / train_coarse_total:.4f}',
        })

    avg_train_loss = train_loss / train_coarse_total
    avg_train_acc  = train_coarse_correct / train_coarse_total

    # ──── 验证 ────────────────────────────────
    model.eval()
    val_loss = 0.0
    coarse_correct = coarse_total = 0
    fine_correct   = fine_total   = 0

    with torch.no_grad():
        for inputs, _, yc, yf in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            yc     = yc.to(device, non_blocking=True)
            yf     = yf.to(device, non_blocking=True)

            with autocast():
                lc_out, lf_out, _ = model(inputs)
                loss = compute_combined_loss(lc_out, lf_out, yc, yf)

            val_loss += loss.item() * inputs.size(0)

            pred_c = lc_out.argmax(1)
            coarse_total   += yc.size(0)
            coarse_correct += (pred_c == yc).sum().item()

            # 细分类：仅统计 QAM 样本（yf != -1）
            qam_mask = (yf != -1)
            if qam_mask.sum() > 0:
                pred_f       = lf_out.argmax(1)
                fine_total   += qam_mask.sum().item()
                fine_correct += (pred_f[qam_mask] == yf[qam_mask]).sum().item()

    avg_val_loss   = val_loss / coarse_total
    avg_val_acc    = coarse_correct / coarse_total           # 粗分类准确率
    avg_fine_acc   = fine_correct / fine_total if fine_total else 0.0

    et = time.time() - t0
    epoch_times.append(et)

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    print(f"  Epoch {epoch+1}: "
          f"TrainCoarse={avg_train_acc:.4f} "
          f"ValCoarse={avg_val_acc:.4f} "
          f"ValFine(QAM)={avg_fine_acc:.4f} "
          f"ValLoss={avg_val_loss:.4f} "
          f"LR={optimizer.param_groups[0]['lr']:.6f} "
          f"[{et:.1f}s]")

    # 以粗+细联合验证指标决定保存
    combined_acc = 0.6 * avg_val_acc + 0.4 * avg_fine_acc
    if combined_acc > best_val_acc:
        print(f"  ✓ Saved best combined acc ({best_val_acc:.4f} → {combined_acc:.4f})")
        torch.save(model.state_dict(), filepath)
        best_val_acc  = combined_acc
        best_val_loss = avg_val_loss
        patience_counter = 0
    elif avg_val_loss < best_val_loss:
        print(f"  ✓ Saved best val loss ({best_val_loss:.4f} → {avg_val_loss:.4f})")
        torch.save(model.state_dict(), filepath)
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch+1}.")
            break

# ─────────────────────────────────────────────
# 训练总结
# ─────────────────────────────────────────────
total_time = np.sum(epoch_times)
print(f"\n{'='*60}")
print(f"[{MODEL_NAME}] Training Summary")
print(f"  Total epochs   : {len(epoch_times)}")
print(f"  Avg epoch time : {np.mean(epoch_times):.1f}s")
print(f"  Total time     : {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"  Best combined  : {best_val_acc:.4f}")
print(f"  Parameters     : {total_params:,}")
print(f"  Weights saved  : {filepath}")
print(f"{'='*60}")

mltools.show_history(history['train_loss'], history['val_loss'],
                     history['train_acc'],  history['val_acc'])


# ─────────────────────────────────────────────
# 测试评估
# ─────────────────────────────────────────────
def evaluate(model, test_loader, device, model_path, classes):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_coarse_pred = []
    all_fine_pred   = []
    all_coarse_gt   = []
    all_fine_gt     = []

    with torch.no_grad():
        for inputs, _, yc, yf in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            with autocast():
                lc_out, lf_out, _ = model(inputs)

            all_coarse_pred.append(lc_out.argmax(1).cpu().numpy())
            all_fine_pred.append(lf_out.argmax(1).cpu().numpy())
            all_coarse_gt.append(yc.numpy())
            all_fine_gt.append(yf.numpy())

    coarse_pred = np.concatenate(all_coarse_pred)
    fine_pred   = np.concatenate(all_fine_pred)
    coarse_gt   = np.concatenate(all_coarse_gt)
    fine_gt     = np.concatenate(all_fine_gt)

    # 粗分类准确率
    coarse_acc = (coarse_pred == coarse_gt).mean()
    print(f"\n{'='*50}")
    print(f"  Stage 1 (Coarse) Accuracy: {coarse_acc*100:.2f}%")
    print(f"  QAM recall    : "
          f"{((coarse_pred == 1) & (coarse_gt == 1)).sum() / (coarse_gt == 1).sum() * 100:.2f}%")
    print(f"  NonQAM recall : "
          f"{((coarse_pred == 0) & (coarse_gt == 0)).sum() / (coarse_gt == 0).sum() * 100:.2f}%")

    # 细分类（仅 QAM 真实样本）
    qam_mask = (fine_gt != -1)
    if qam_mask.sum() > 0:
        fine_acc = (fine_pred[qam_mask] == fine_gt[qam_mask]).mean()
        print(f"\n  Stage 2 (Fine, QAM only) Accuracy: {fine_acc*100:.2f}%")
        print(f"  Per-class:")
        for i, name in enumerate(QAM_CLASSES):
            mask_i = (fine_gt == i)
            if mask_i.sum() > 0:
                cls_acc = (fine_pred[mask_i] == i).sum() / mask_i.sum()
                print(f"    {name:>8s}: {cls_acc*100:.2f}%  ({mask_i.sum()} samples)")

    # 两阶段联合准确率（Stage1 正确判断为 QAM 且 Stage2 也正确）
    both_correct = qam_mask & (coarse_pred == 1) & (fine_pred == fine_gt)
    joint_acc = both_correct.sum() / qam_mask.sum() if qam_mask.sum() > 0 else 0
    print(f"\n  End-to-End QAM accuracy (Stage1∩Stage2): {joint_acc*100:.2f}%")
    print(f"{'='*50}")


print(f"\n--- [{MODEL_NAME}] Evaluation ---")
evaluate(model, test_loader, device, filepath, classes)
print("Done.")
