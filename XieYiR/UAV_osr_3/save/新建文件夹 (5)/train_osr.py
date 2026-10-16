import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from models import STFTSplitDataset, FullFeatureExtractor, SubCenterArcMarginProduct

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. 数据划分与透明审计
# ==========================================
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

    print("\n" + "="*60)
    print("           DATASET PARTITION TRANSPARENCY AUDIT")
    print("="*60)

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

    print("-" * 60)
    print(f"Total: Train={len(train_files)}, Val={len(val_files)}, Test Known={len(test_known_files)}, Test Unknown={len(test_unknown_files)}")
    print("="*60 + "\n")

    return (train_files, train_labels), (val_files, val_labels), \
           (test_known_files, test_known_labels), (test_unknown_files, test_unknown_labels), known_map

# ==========================================
# 2. 训练引擎 (适配 SubCenter-ArcFace)
# ==========================================
def train_model(extractor, head, train_loader, val_loader, device, epochs=100, patience=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(extractor.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_weights_path = "best_subcenter_osr_model.pt"

    print("-> Start Training Sub-Center ArcFace Model (K=3)...")
    for epoch in range(1, epochs + 1):
        cur_m = min(head.m, head.m * (epoch / 5.0))

        extractor.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            feats = extractor(x)
            margin_logits, raw_cosine = head(feats, y, current_margin=cur_m)
            loss = criterion(margin_logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            pred = torch.argmax(raw_cosine, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()

        # 验证
        extractor.eval()
        head.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                feats = extractor(x)
                margin_logits, raw_cosine = head(feats, y, current_margin=cur_m)
                loss = criterion(margin_logits, y)
                val_loss += loss.item() * len(y)
                pred = torch.argmax(raw_cosine, dim=1)
                val_correct += (pred == y).sum().item()
                val_total += len(y)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch:03d}/{epochs:03d}] (m: {cur_m:.3f}) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:6.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:6.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({'extractor': extractor.state_dict(), 'head': head.state_dict()}, best_weights_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement for {patience} consecutive epochs.")
                break

    print(f"Loading best weights from {best_weights_path} (Best Val Loss: {best_val_loss:.4f})")
    checkpoint = torch.load(best_weights_path)
    extractor.load_state_dict(checkpoint['extractor'])
    head.load_state_dict(checkpoint['head'])
    return extractor, head

# ==========================================
# 3. 拒绝阈值标定
# ==========================================
@torch.no_grad()
def calibrate_rejection_threshold(extractor, head, val_loader, device):
    extractor.eval()
    head.eval()
    sim_scores = []
    
    for x, y, _ in val_loader:
        x, y = x.to(device), y.to(device)
        feats = extractor(x)
        cos_sims = head(feats, label=None)
        max_sim, preds = torch.max(cos_sims, dim=1)
        
        correct_mask = (preds == y)
        if correct_mask.sum() > 0:
            sim_scores.extend(max_sim[correct_mask].cpu().numpy().tolist())

    sim_scores = np.array(sim_scores)
    # 取验证集正确样本后 10% 的分位数，确保 90% 的已知类召回
    tau = float(np.percentile(sim_scores, 10))
    print(f"\n[Threshold Calibration] Sub-Center Cosine Sim Stats -> Mean: {np.mean(sim_scores):.4f}, Min: {np.min(sim_scores):.4f}, Max: {np.max(sim_scores):.4f}")
    print(f"[Threshold Calibration] Calibrated Rejection Threshold (tau): {tau:.4f}")
    return tau

# ==========================================
# 4. 开放集识别与全景混淆矩阵
# ==========================================
@torch.no_grad()
def evaluate_open_set(extractor, head, known_loader, unknown_loader, threshold, known_map, unknown_classes, device):
    extractor.eval()
    head.eval()
    idx_to_known = {v: k for k, v in known_map.items()}
    UNKNOWN_LABEL_ID = len(known_map)

    # 1. 评估已知类
    known_true, known_pred = [], []
    class_sims = {c: [] for c in known_map.keys()}

    for x, y, _ in known_loader:
        x = x.to(device)
        feats = extractor(x)
        cos_sims = head(feats, label=None)
        max_sim, preds = torch.max(cos_sims, dim=1)

        for true_idx, sim_val in zip(y.numpy(), max_sim.cpu().numpy()):
            class_sims[idx_to_known[true_idx]].append(sim_val)

        preds_osr = torch.where(max_sim >= threshold, preds, torch.tensor(UNKNOWN_LABEL_ID).to(device))
        known_true.extend(y.numpy())
        known_pred.extend(preds_osr.cpu().numpy())

    known_true = np.array(known_true)
    known_pred = np.array(known_pred)

    print("\n[Known Classes Test Sim Distribution (Sub-Centers)]")
    for c_name, sims in class_sims.items():
        if len(sims) > 0:
            print(f"  Class [{c_name}] -> Avg Sim: {np.mean(sims):.4f} | Min: {np.min(sims):.4f} | Max: {np.max(sims):.4f}")

    # 2. 评估未知类
    unknown_results = {u_cls: {'total': 0, 'correct_unknown': 0, 'misclassified_as': {k: 0 for k in known_map.keys()}} 
                       for u_cls in sorted(unknown_classes)}

    for x, orig_labels, _ in unknown_loader:
        x = x.to(device)
        feats = extractor(x)
        cos_sims = head(feats, label=None)
        max_sim, preds = torch.max(cos_sims, dim=1)
        preds_osr = torch.where(max_sim >= threshold, preds, torch.tensor(UNKNOWN_LABEL_ID).to(device)).cpu().numpy()

        for u_cls, pred_id in zip(orig_labels, preds_osr):
            unknown_results[u_cls]['total'] += 1
            if pred_id == UNKNOWN_LABEL_ID:
                unknown_results[u_cls]['correct_unknown'] += 1
            else:
                pred_class_name = idx_to_known[pred_id]
                unknown_results[u_cls]['misclassified_as'][pred_class_name] += 1

    # 3. 统计指标报告
    closed_set_acc = np.mean(known_true == (known_pred % UNKNOWN_LABEL_ID))
    known_osr_acc = np.mean(known_true == known_pred)
    known_misreject_rate = np.mean(known_pred == UNKNOWN_LABEL_ID)
    total_unknown_samples = sum(r['total'] for r in unknown_results.values())
    total_correct_unknown = sum(r['correct_unknown'] for r in unknown_results.values())

    print("\n" + "="*65)
    print("           OPEN SET RECOGNITION (OSR) EVALUATION REPORT")
    print("="*65)
    print(f"Rejection Threshold (tau): {threshold:.4f}")
    print(f"1. Known Class Metrics:")
    print(f"   - Standard Closed-set Top-1 Acc:  {closed_set_acc * 100:.2f}%")
    print(f"   - Open-set Known Accuracy (Acc_K):{known_osr_acc * 100:.2f}%")
    print(f"   - Known False Rejection Rate:     {known_misreject_rate * 100:.2f}%")
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

# ==========================================
# 5. 字符表打印与热力图 (已修复反斜杠兼容语法)
# ==========================================
def plot_and_print_confusion_matrix(known_true, known_pred, unknown_results, known_map, UNKNOWN_ID, save_path="subcenter_osr_confusion_matrix.png"):
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

    # 1. 终端纯文本输出
    print("FULL OPEN-SET CONFUSION MATRIX TABLE (Rows: True Class, Cols: Predicted Class):")
    header = "True/Pred".ljust(12) + "".join([f"{col:<10}" for col in col_labels])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r_idx, r_name in enumerate(row_labels):
        row_str = f"{r_name:<12}" + "".join([f"{cm[r_idx, c_idx]:<10}" for c_idx in range(len(col_labels))])
        print(row_str)
    print("-" * len(header) + "\n")

    # 2. 图像保存
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=col_labels, yticklabels=row_labels, cbar=True)
    plt.title("Sub-Center Open-Set Confusion Matrix (Knowns + Unknowns)", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Open set recognition confusion matrix saved to: {save_path}")
    plt.show()

# ==========================================
# 6. 主程序入口
# ==========================================
def main():
    TRAIN_DIR = "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EPOCHS = 100
    PATIENCE = 10
    LR = 1e-4

    set_seed(42)

    KNOWN_CLASSES = ['A', 'D', 'E', 'G']
    UNKNOWN_CLASSES = ['B', 'C', 'F']
    K_SUB_CENTERS = 3  # 每个已知类分配 3 个子中心

    tr_data, val_data, te_known, te_unknown, known_map = split_datasets_by_files(
        TRAIN_DIR, KNOWN_CLASSES, UNKNOWN_CLASSES
    )

    train_ds  = STFTSplitDataset(tr_data[0], tr_data[1])
    val_ds    = STFTSplitDataset(val_data[0], val_data[1])
    test_k_ds = STFTSplitDataset(te_known[0], te_known[1])
    test_u_ds = STFTSplitDataset(te_unknown[0], te_unknown[1])

    train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader    = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_k_loader = DataLoader(test_k_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_u_loader = DataLoader(test_u_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 骨干网络 (win_size=32, stride=16)
    extractor = FullFeatureExtractor(win_size=32, stride=16, d_model=512).to(DEVICE)
    
    # 实例化 SubCenterArcMarginProduct 头，设置 K=3
    subcenter_head = SubCenterArcMarginProduct(
        in_features=512, out_features=len(KNOWN_CLASSES), K=K_SUB_CENTERS, s=20.0, m=0.35
    ).to(DEVICE)

    # 训练模型
    extractor, subcenter_head = train_model(
        extractor, subcenter_head, train_loader, val_loader, 
        device=DEVICE, epochs=EPOCHS, patience=PATIENCE, lr=LR
    )

    # 标定拒绝阈值
    threshold = calibrate_rejection_threshold(extractor, subcenter_head, val_loader, device=DEVICE)

    # 开放集评估
    evaluate_open_set(
        extractor, subcenter_head, 
        test_k_loader, test_u_loader, 
        threshold=threshold, known_map=known_map, 
        unknown_classes=UNKNOWN_CLASSES, device=DEVICE
    )

if __name__ == "__main__":
    main()