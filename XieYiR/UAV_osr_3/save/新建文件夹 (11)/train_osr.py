import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns

from models import STFTSplitDataset, GlobalSpectrogramResNet, PrototypeLayer, PrototypeLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_datasets_by_files(train_dir, known_classes=['A', 'D', 'E', 'G'], unknown_classes=['B', 'C', 'F']):
    all_files = glob.glob(os.path.join(train_dir, "*.h5"))
    if len(all_files) == 0:
        raise ValueError(f"No .h5 files found in {train_dir}!")

    class_to_files = {}
    for f in all_files:
        c = os.path.basename(f)[0].upper()
        class_to_files.setdefault(c, []).append(f)

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_known_files, test_known_labels = [], []
    test_unknown_files, test_unknown_labels = [], []

    known_map = {c: i for i, c in enumerate(sorted(known_classes))}

    print("\n" + "="*65)
    print("      DATASET PARTITION (PROTOTYPE COMPACT LEARNING)")
    print("="*65)

    for c in known_classes:
        files = class_to_files.get(c, [])
        random.shuffle(files)
        n = len(files)
        if n == 0:
            raise ValueError(f"Class [{c}] has 0 files! Check filenames.")

        n_train = int(n * 0.8)
        n_val = max(1, int(round(n * 0.1)))
        n_test = n - n_train - n_val

        c_train = files[:n_train]
        c_val = files[n_train:n_train + n_val]
        c_test = files[n_train + n_val:]

        train_files.extend(c_train)
        train_labels.extend([known_map[c]] * len(c_train))
        val_files.extend(c_val)
        val_labels.extend([known_map[c]] * len(c_val))
        test_known_files.extend(c_test)
        test_known_labels.extend([known_map[c]] * len(c_test))

        print(f"Known Class   [{c}] -> Total: {n:<4} | Train: {len(c_train):<4} | Val: {len(c_val):<3} | Test: {len(c_test):<3}")

    for c in unknown_classes:
        files = class_to_files.get(c, [])
        test_unknown_files.extend(files)
        test_unknown_labels.extend([c] * len(files))
        print(f"Unknown Class [{c}] -> Total: {len(files):<4} (100% into Test)")

    print("-" * 65)
    print(f"Total: Train={len(train_files)}, Val={len(val_files)}, Test Known={len(test_known_files)}, Test Unknown={len(test_unknown_files)}")
    print("="*65 + "\n")

    return (train_files, train_labels), (val_files, val_labels), \
           (test_known_files, test_known_labels), (test_unknown_files, test_unknown_labels), known_map

def train_model(model, proto_layer, criterion, train_loader, val_loader, device, epochs=100, patience=10, lr=2e-4):
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(proto_layer.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    best_weights_path = "best_proto_model.pt"

    print("-> Start Training Compact Prototype Network...")
    for epoch in range(1, epochs + 1):
        model.train()
        proto_layer.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                feats = model(x)
                logits, dist = proto_layer(feats)
                loss = criterion(logits, dist, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()

        # 验证阶段
        model.eval()
        proto_layer.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast():
                    feats = model(x)
                    logits, dist = proto_layer(feats)
                    loss = criterion(logits, dist, y)

                val_loss += loss.item() * len(y)
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == y).sum().item()
                val_total += len(y)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:6.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:6.2f}%")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({'model': model.state_dict(), 'proto_layer': proto_layer.state_dict()}, best_weights_path)
            saved_tag = " [*SAVED BEST*]"
        else:
            early_stop_counter += 1
            saved_tag = ""

        if saved_tag:
            print(f"           {saved_tag}")

        if early_stop_counter >= patience:
            print(f"\n[Early Stopping Triggered] No improvement for {patience} consecutive epochs.")
            break

    checkpoint = torch.load(best_weights_path)
    model.load_state_dict(checkpoint['model'])
    proto_layer.load_state_dict(checkpoint['proto_layer'])
    return model, proto_layer

# =========================================================================
# 核心革命改动：为每个已知类独立标定欧氏半径 R_k (彻底按死单邻居漏洞)
# =========================================================================
@torch.no_grad()
def calibrate_class_specific_thresholds(model, proto_layer, val_loader, num_classes, device, percentile=90):
    model.eval()
    proto_layer.eval()
    class_distances = {i: [] for i in range(num_classes)}

    for x, y, _ in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            feats = model(x)
            _, dist = proto_layer(feats) # dist: [B, num_classes]

        # 仅收集分类正确的样本到自身类原型的距离
        preds = torch.argmin(dist, dim=1)
        for i in range(len(y)):
            if preds[i] == y[i]:
                d = dist[i, y[i]].item()
                class_distances[y[i].item()].append(d)

    # 计算各类别的专属截断半径 (覆盖该类 90% 的已知验证样本)
    class_radii = {}
    print("\n" + "="*50)
    print("      CALIBRATED CLASS-SPECIFIC RADII (R_k)")
    print("="*50)
    for c_id in range(num_classes):
        dists = np.array(class_distances[c_id])
        if len(dists) > 0:
            r_k = float(np.percentile(dists, percentile))
        else:
            r_k = 1.0 # 默认安全距离
        class_radii[c_id] = r_k
        print(f"Class [{c_id}] Max Accepted Distance Radius: {r_k:.4f}")
    print("="*50 + "\n")
    return class_radii

# ==========================================
# 开放集评估 (基于绝对球体拦截)
# ==========================================
@torch.no_grad()
def evaluate_open_set(model, proto_layer, known_loader, unknown_loader, class_radii, known_map, unknown_classes, device):
    model.eval()
    proto_layer.eval()
    idx_to_known = {v: k for k, v in known_map.items()}
    UNKNOWN_LABEL_ID = len(known_map)

    known_true, known_pred = [], []
    for x, y, _ in known_loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            feats = model(x)
            _, dist = proto_layer(feats)

        # 找到最近的类原型
        min_dists, preds = torch.min(dist, dim=1)
        
        # 绝对物理球门限判定：
        preds_osr = []
        for d_val, pred_cls in zip(min_dists, preds):
            pred_id = pred_cls.item()
            # 只有当样本到预测原型的距离 <= 该类专属半径时，才承认是已知类
            if d_val.item() <= class_radii[pred_id]:
                preds_osr.append(pred_id)
            else:
                preds_osr.append(UNKNOWN_LABEL_ID)

        known_true.extend(y.numpy())
        known_pred.extend(preds_osr)

    known_true = np.array(known_true)
    known_pred = np.array(known_pred)

    unknown_results = {u_cls: {'total': 0, 'correct_unknown': 0, 'misclassified_as': {k: 0 for k in known_map.keys()}} 
                       for u_cls in sorted(unknown_classes)}

    for x, orig_labels, _ in unknown_loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            feats = model(x)
            _, dist = proto_layer(feats)

        min_dists, preds = torch.min(dist, dim=1)

        preds_osr = []
        for d_val, pred_cls in zip(min_dists, preds):
            pred_id = pred_cls.item()
            # 未知类即使离 E 最近，只要其绝对距离突破了 E 的球体，立即判定为 UNKNOWN！
            if d_val.item() <= class_radii[pred_id]:
                preds_osr.append(pred_id)
            else:
                preds_osr.append(UNKNOWN_LABEL_ID)

        for u_cls, pred_id in zip(orig_labels, preds_osr):
            unknown_results[u_cls]['total'] += 1
            if pred_id == UNKNOWN_LABEL_ID:
                unknown_results[u_cls]['correct_unknown'] += 1
            else:
                pred_class_name = idx_to_known[pred_id]
                unknown_results[u_cls]['misclassified_as'][pred_class_name] += 1

    closed_set_acc = np.mean(known_true == (known_pred % UNKNOWN_LABEL_ID))
    known_osr_acc = np.mean(known_true == known_pred)
    total_unknown_samples = sum(r['total'] for r in unknown_results.values())
    total_correct_unknown = sum(r['correct_unknown'] for r in unknown_results.values())

    print("\n" + "="*65)
    print("      PROTOTYPE CLASS-CONDITIONAL OSR REPORT")
    print("="*65)
    print(f"1. Known Class Metrics:")
    print(f"   - Standard Closed-set Top-1 Acc:  {closed_set_acc * 100:.2f}%")
    print(f"   - Open-set Known Accuracy (Acc_K):{known_osr_acc * 100:.2f}%")
    print(f"\n2. Unknown Class Metrics:")
    print(f"   - Overall Unknown Rejection Rate: {total_correct_unknown / total_unknown_samples * 100:.2f}%")
    print("-" * 65)
    print("3. Fine-grained Unknown Breakdown:")
    for u_cls, info in unknown_results.items():
        tot, corr = info['total'], info['correct_unknown']
        rec_rate = (corr / tot * 100) if tot > 0 else 0.0
        mis = [f"{k}: {v}" for k, v in info['misclassified_as'].items() if v > 0]
        mis_str = ", ".join(mis) if mis else "None (100% Rejected)"
        print(f"   * Unknown [{u_cls}] (Total: {tot:<3}) -> Rejected: {corr:<3} ({rec_rate:6.2f}%) | Misclassified Into: {mis_str}")
    print("="*65 + "\n")

    plot_and_print_confusion_matrix(known_true, known_pred, unknown_results, known_map, UNKNOWN_LABEL_ID)

def plot_and_print_confusion_matrix(known_true, known_pred, unknown_results, known_map, UNKNOWN_ID, save_path="proto_osr_confusion_matrix.png"):
    col_labels = [k for k, _ in sorted(known_map.items(), key=lambda item: item[1])] + ['Unknown']
    sorted_unknown_keys = sorted(list(unknown_results.keys()))
    row_labels = col_labels[:-1] + sorted_unknown_keys

    cm = np.zeros((len(row_labels), len(col_labels)), dtype=np.int32)
    for t, p in zip(known_true, known_pred):
        cm[t, p] += 1
    for i, u_cls in enumerate(sorted_unknown_keys):
        row_idx = len(known_map) + i
        info = unknown_results[u_cls]
        cm[row_idx, UNKNOWN_ID] = info['correct_unknown']
        for k_name, count in info['misclassified_as'].items():
            k_id = known_map[k_name]
            cm[row_idx, k_id] = count

    print("FULL OPEN-SET CONFUSION MATRIX TABLE (Rows: True, Cols: Pred):")
    header = "True/Pred".ljust(12) + "".join([f"{col:<10}" for col in col_labels])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r_idx, r_name in enumerate(row_labels):
        row_str = f"{r_name:<12}" + "".join([f"{cm[r_idx, c_idx]:<10}" for c_idx in range(len(col_labels))])
        print(row_str)
    print("-" * len(header) + "\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=col_labels, yticklabels=row_labels, cbar=True)
    plt.title("Class-Conditional Prototype OSR Confusion Matrix", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Confusion matrix saved to: {save_path}")
    plt.show()

def main():
    # 换回最初你提取的纯净 STFT 路径即可
    TRAIN_DIR = "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 8
    EPOCHS = 100
    PATIENCE = 10
    LR = 2e-4

    set_seed(42)

    KNOWN_CLASSES = ['A', 'D', 'E', 'G']
    UNKNOWN_CLASSES = ['B', 'C', 'F']

    tr_data, val_data, te_known, te_unknown, known_map = split_datasets_by_files(
        TRAIN_DIR, KNOWN_CLASSES, UNKNOWN_CLASSES
    )

    train_ds  = STFTSplitDataset(tr_data[0], tr_data[1])
    val_ds    = STFTSplitDataset(val_data[0], val_data[1])
    test_k_ds = STFTSplitDataset(te_known[0], te_known[1])
    test_u_ds = STFTSplitDataset(te_unknown[0], te_unknown[1])

    train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader    = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=2, pin_memory=True, persistent_workers=True)
    test_k_loader = DataLoader(test_k_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_u_loader = DataLoader(test_u_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 实例化紧凑原型模型
    model = GlobalSpectrogramResNet(in_channels=2, feat_dim=256).to(DEVICE)
    proto_layer = PrototypeLayer(num_classes=len(KNOWN_CLASSES), feat_dim=256, scale=10.0).to(DEVICE)
    criterion = PrototypeLoss(weight_compact=0.5).to(DEVICE)

    # 训练
    model, proto_layer = train_model(
        model, proto_layer, criterion, train_loader, val_loader, 
        device=DEVICE, epochs=EPOCHS, patience=PATIENCE, lr=LR
    )

    # 核心：单独为每个已知类标定排他性半径 R_k
    class_radii = calibrate_class_specific_thresholds(
        model, proto_layer, val_loader, num_classes=len(KNOWN_CLASSES), device=DEVICE, percentile=90
    )

    # 评测
    evaluate_open_set(
        model, proto_layer, 
        test_k_loader, test_u_loader, 
        class_radii=class_radii, known_map=known_map, 
        unknown_classes=UNKNOWN_CLASSES, device=DEVICE
    )

if __name__ == "__main__":
    main()