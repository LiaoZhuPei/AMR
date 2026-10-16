import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns

from models import STFTSplitDataset, DTS_SpectrogramResNet, ArcMarginProduct

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================================================================
# 1. 严格文件级划分
# =========================================================================
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
    print("      DATASET PARTITION (DTS-RESNET DUAL-STREAM OSR)")
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

# =========================================================================
# 2. 训练引擎
# =========================================================================
def train_model(model, arc_layer, criterion, train_loader, val_loader, device, epochs=100, patience=15, lr=2e-4):
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(arc_layer.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0
    best_weights_path = "best_dts_arcface_model.pt"

    print("-> Start Training DTS-ResNet (Dual-Stream) with ArcFace...")
    for epoch in range(1, epochs + 1):
        model.train()
        arc_layer.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                feats = model(x)
                logits = arc_layer(feats, y)
                loss = criterion(logits, y)

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
        arc_layer.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast():
                    feats = model(x)
                    cos_sim = arc_layer(feats, label=None)
                    loss = criterion(cos_sim * arc_layer.s, y)

                val_loss += loss.item() * len(y)
                pred = torch.argmax(cos_sim, dim=1)
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
            torch.save({'model': model.state_dict(), 'arc_layer': arc_layer.state_dict()}, best_weights_path)
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
    arc_layer.load_state_dict(checkpoint['arc_layer'])
    return model, arc_layer

# =========================================================================
# 3. 稳健高斯门限标定 (alpha=1.0)
# =========================================================================
@torch.no_grad()
def calibrate_robust_thresholds(model, arc_layer, val_loader, num_classes, device, alpha=1.0):
    model.eval()
    arc_layer.eval()
    class_cosines = {i: [] for i in range(num_classes)}

    for x, y, _ in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            feats = model(x)
            cos_sim = arc_layer(feats, label=None)

        preds = torch.argmax(cos_sim, dim=1)
        for i in range(len(y)):
            if preds[i] == y[i]:
                class_cosines[y[i].item()].append(cos_sim[i, y[i]].item())

    class_thresholds = {}
    print("\n" + "="*65)
    print(f"      ROBUST GAUSSIAN THRESHOLDS (tau = mean - {alpha}*std)")
    print("="*65)
    for c_id in range(num_classes):
        sims = np.array(class_cosines[c_id])
        if len(sims) > 1:
            mu = float(np.mean(sims))
            sigma = float(np.std(sims))
            tau_k = mu - alpha * sigma
        else:
            tau_k = 0.80
            mu, sigma = 0.88, 0.03
        
        class_thresholds[c_id] = tau_k
        print(f"Class [{c_id}] Stats: Mean={mu:.4f}, Std={sigma:.4f} -> Calibrated Tau: {tau_k:.4f}")
    print("="*65 + "\n")
    return class_thresholds

# =========================================================================
# 4. 深度透视与开放集评估
# =========================================================================
@torch.no_grad()
def evaluate_open_set_diagnostic(model, arc_layer, known_loader, unknown_loader, class_thresholds, known_map, unknown_classes, device):
    model.eval()
    arc_layer.eval()
    idx_to_known = {v: k for k, v in known_map.items()}
    e_class_id = known_map['E']
    UNKNOWN_LABEL_ID = len(known_map)

    e_known_sims = []
    known_true, known_pred = [], []
    for x, y, _ in known_loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            feats = model(x)
            cos_sim = arc_layer(feats, label=None)

        max_sims, preds = torch.max(cos_sim, dim=1)
        
        preds_osr = []
        for i, (sim_val, pred_cls) in enumerate(zip(max_sims, preds)):
            pred_id = pred_cls.item()
            if y[i].item() == e_class_id:
                e_known_sims.append(cos_sim[i, e_class_id].item())

            if sim_val.item() >= class_thresholds[pred_id]:
                preds_osr.append(pred_id)
            else:
                preds_osr.append(UNKNOWN_LABEL_ID)

        known_true.extend(y.numpy())
        known_pred.extend(preds_osr)

    known_true = np.array(known_true)
    known_pred = np.array(known_pred)

    c_unknown_to_e_sims = []
    unknown_results = {u_cls: {'total': 0, 'correct_unknown': 0, 'misclassified_as': {k: 0 for k in known_map.keys()}} 
                       for u_cls in sorted(unknown_classes)}

    for x, orig_labels, _ in unknown_loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            feats = model(x)
            cos_sim = arc_layer(feats, label=None)

        max_sims, preds = torch.max(cos_sim, dim=1)

        preds_osr = []
        for i, (sim_val, pred_cls) in enumerate(zip(max_sims, preds)):
            pred_id = pred_cls.item()
            u_label = orig_labels[i]

            if u_label == 'C':
                c_unknown_to_e_sims.append(cos_sim[i, e_class_id].item())

            if sim_val.item() >= class_thresholds[pred_id]:
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

    # 深度诊断透视
    print("\n" + "#"*65)
    print("      DEEP TELEMETRY: KNOWN [E] vs UNKNOWN [C] COSINE SIMILARITY")
    print("#"*65)
    e_known_sims = np.array(e_known_sims)
    c_unknown_to_e_sims = np.array(c_unknown_to_e_sims)
    tau_e = class_thresholds[e_class_id]

    print(f"Current Class [E] Decision Threshold (tau_E): {tau_e:.4f}")
    if len(e_known_sims) > 0:
        print(f"True Known [E] -> Min: {np.min(e_known_sims):.4f} | Mean: {np.mean(e_known_sims):.4f} | Max: {np.max(e_known_sims):.4f}")
    if len(c_unknown_to_e_sims) > 0:
        print(f"Unknown    [C] -> Min: {np.min(c_unknown_to_e_sims):.4f} | Mean: {np.mean(c_unknown_to_e_sims):.4f} | Max: {np.max(c_unknown_to_e_sims):.4f}")
        c_above_tau = np.sum(c_unknown_to_e_sims >= tau_e)
        print(f"Diagnostic Result: There are {c_above_tau}/{len(c_unknown_to_e_sims)} C samples whose cosine >= tau_E!")
    print("#"*65 + "\n")

    closed_set_acc = np.mean(known_true == (known_pred % UNKNOWN_LABEL_ID))
    known_osr_acc = np.mean(known_true == known_pred)
    total_unknown_samples = sum(r['total'] for r in unknown_results.values())
    total_correct_unknown = sum(r['correct_unknown'] for r in unknown_results.values())

    print("\n" + "="*65)
    print("      DTS-RESNET DUAL-STREAM OSR FINAL REPORT")
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

def plot_and_print_confusion_matrix(known_true, known_pred, unknown_results, known_map, UNKNOWN_ID, save_path="dts_osr_confusion_matrix.png"):
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
    plt.title("DTS-ResNet Dual-Stream OSR Confusion Matrix", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Confusion matrix saved to: {save_path}")
    plt.show()

# =========================================================================
# 5. 主程序
# =========================================================================
def main():
    TRAIN_DIR = "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 8
    EPOCHS = 100
    PATIENCE = 15
    LR = 2e-4
    ALPHA = 1.0

    set_seed(42)

    KNOWN_CLASSES = ['A', 'D', 'E', 'G']
    UNKNOWN_CLASSES = ['B', 'C', 'F']

    tr_data, val_data, te_known, te_unknown, known_map = split_datasets_by_files(
        TRAIN_DIR, KNOWN_CLASSES, UNKNOWN_CLASSES
    )

    train_ds  = STFTSplitDataset(tr_data[0], tr_data[1], is_train=True)
    val_ds    = STFTSplitDataset(val_data[0], val_data[1], is_train=False)
    test_k_ds = STFTSplitDataset(te_known[0], te_known[1], is_train=False)
    test_u_ds = STFTSplitDataset(te_unknown[0], te_unknown[1], is_train=False)

    train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader    = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                               num_workers=2, pin_memory=True, persistent_workers=True)
    test_k_loader = DataLoader(test_k_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_u_loader = DataLoader(test_u_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 实例化时-频双流网络与稳健 ArcFace (s=32.0, m=0.50)
    model = DTS_SpectrogramResNet(in_channels=2, feat_dim=512).to(DEVICE)
    arc_layer = ArcMarginProduct(in_features=512, out_features=len(KNOWN_CLASSES), s=32.0, m=0.50).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # 训练
    model, arc_layer = train_model(
        model, arc_layer, criterion, train_loader, val_loader, 
        device=DEVICE, epochs=EPOCHS, patience=PATIENCE, lr=LR
    )

    # 稳健门限标定
    class_thresholds = calibrate_robust_thresholds(
        model, arc_layer, val_loader, num_classes=len(KNOWN_CLASSES), device=DEVICE, alpha=ALPHA
    )

    # 诊断评估
    evaluate_open_set_diagnostic(
        model, arc_layer, 
        test_k_loader, test_u_loader, 
        class_thresholds=class_thresholds, known_map=known_map, 
        unknown_classes=UNKNOWN_CLASSES, device=DEVICE
    )

if __name__ == "__main__":
    main()