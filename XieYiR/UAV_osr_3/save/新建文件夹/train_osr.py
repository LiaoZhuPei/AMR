import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models import STFTSplitDataset, FullFeatureExtractor, ArcMarginProduct

# 固定随机种子确保切分与结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. 数据划分逻辑: 文件级 8:1:1 严格物理隔离
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

    # (1) 已知类: 单独按 8:1:1 划分
    for c in known_classes:
        files = class_to_files.get(c, [])
        random.shuffle(files)
        n = len(files)
        if n < 3:
            raise ValueError(f"Class {c} has too few files ({n}) to split 8:1:1.")

        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        if n_val == 0: n_val = 1
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

    # (2) 未知类: 100% 划入测试集，完全不参与训练与验证
    for c in unknown_classes:
        files = class_to_files.get(c, [])
        test_unknown_files.extend(files)
        test_unknown_labels.extend([c] * len(files)) # 保留真实字符标识用于后续细粒度统计

    print(f"[Dataset Partition Summary]")
    print(f"  Known Classes: {known_classes} (Indices: {known_map})")
    print(f"  Unknown Classes: {unknown_classes}")
    print(f"  Train Known Files: {len(train_files)}")
    print(f"  Val Known Files:   {len(val_files)}")
    print(f"  Test Known Files:  {len(test_known_files)}")
    print(f"  Test Unknown Files:{len(test_unknown_files)}")

    return (train_files, train_labels), (val_files, val_labels), \
           (test_known_files, test_known_labels), (test_unknown_files, test_unknown_labels), known_map

# ==========================================
# 2. 训练与早停引擎
# ==========================================
def train_model(extractor, head, train_loader, val_loader, device, epochs=100, patience=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(extractor.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    early_stop_counter = 0
    best_weights_path = "best_osr_model.pt"

    print("\n-> Start Training Metric Learning Model (ArcFace)...")
    for epoch in range(1, epochs + 1):
        extractor.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            feats = extractor(x)
            logits = head(feats, y)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()

        # 验证阶段
        extractor.eval()
        head.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                feats = extractor(x)
                logits = head(feats, y)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == y).sum().item()
                val_total += len(y)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        # 10 代无提升触发早停
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
# 3. 阈值自适应标定 (基于验证集余弦相似度分布)
# ==========================================
@torch.no_grad()
def calibrate_rejection_threshold(extractor, head, val_loader, device, target_recall=0.95):
    """
    通过在已知验证集上统计最大余弦相似度，将对应分位数设为阈值，
    保证在已知类上保持约 target_recall (95%) 的接受率。
    """
    extractor.eval()
    head.eval()
    max_sims = []
    for x, y, _ in val_loader:
        x, y = x.to(device), y.to(device)
        feats = extractor(x)
        cos_sims = head(feats, label=None)  # [B, 4] 余弦相似度
        max_sim, preds = torch.max(cos_sims, dim=1)
        # 仅统计预测正确的已知类样本得分
        correct_mask = (preds == y)
        if correct_mask.sum() > 0:
            max_sims.extend(max_sim[correct_mask].cpu().numpy().tolist())

    max_sims = np.array(max_sims)
    # 取 (1 - target_recall) 的分位数作为拒绝阈值
    threshold = np.percentile(max_sims, (1.0 - target_recall) * 100)
    print(f"\n[Threshold Calibration] Selected Cosine Similarity Threshold: {threshold:.4f} (Target Known TPR: {target_recall*100}%)")
    return float(threshold)

# ==========================================
# 4. 开放集测试与细粒度指标统计
# ==========================================
@torch.no_grad()
def evaluate_open_set(extractor, head, known_loader, unknown_loader, threshold, known_map, unknown_classes, device):
    extractor.eval()
    head.eval()
    idx_to_known = {v: k for k, v in known_map.items()}
    UNKNOWN_LABEL_ID = len(known_map) # 未知类预测标识

    # (1) 测试已知类别
    known_true, known_pred = [], []
    for x, y, _ in known_loader:
        x = x.to(device)
        feats = extractor(x)
        cos_sims = head(feats, label=None)
        max_sim, preds = torch.max(cos_sims, dim=1)
        
        preds_osr = torch.where(max_sim >= threshold, preds, torch.tensor(UNKNOWN_LABEL_ID).to(device))
        known_true.extend(y.numpy())
        known_pred.extend(preds_osr.cpu().numpy())

    known_true = np.array(known_true)
    known_pred = np.array(known_pred)

    closed_set_acc = np.mean(known_true == (known_pred % UNKNOWN_LABEL_ID)) # 闭集强制分类正确率
    known_osr_acc = np.mean(known_true == known_pred)                      # 开放集正确接收且分类正确率
    known_misreject_rate = np.mean(known_pred == UNKNOWN_LABEL_ID)        # 被误判为未知的比例

    # (2) 测试未知类别
    unknown_results = {u_cls: {'total': 0, 'correct_unknown': 0, 'misclassified_as': {k: 0 for k in known_map.keys()}} 
                       for u_cls in unknown_classes}
    all_unknown_pred = []
    for x, orig_labels, _ in unknown_loader:
        x = x.to(device)
        feats = extractor(x)
        cos_sims = head(feats, label=None)
        max_sim, preds = torch.max(cos_sims, dim=1)
        
        preds_osr = torch.where(max_sim >= threshold, preds, torch.tensor(UNKNOWN_LABEL_ID).to(device)).cpu().numpy()
        all_unknown_pred.extend(preds_osr)

        for u_cls, pred_id in zip(orig_labels, preds_osr):
            unknown_results[u_cls]['total'] += 1
            if pred_id == UNKNOWN_LABEL_ID:
                unknown_results[u_cls]['correct_unknown'] += 1
            else:
                pred_class_name = idx_to_known[pred_id]
                unknown_results[u_cls]['misclassified_as'][pred_class_name] += 1

    # (3) 打印全面评估报告
    print("\n" + "="*65)
    print("           OPEN SET RECOGNITION (OSR) EVALUATION REPORT")
    print("="*65)
    print(f"Rejection Threshold (tau): {threshold:.4f}")
    print(f"1. Known Class Metrics (Closed-set Baseline):")
    print(f"   - Standard Closed-set Top-1 Acc:  {closed_set_acc * 100:.2f}%")
    print(f"   - Open-set Known Accuracy (Acc_K):{known_osr_acc * 100:.2f}%")
    print(f"   - Known False Rejection Rate:     {known_misreject_rate * 100:.2f}%")
    
    total_unknown_samples = sum(r['total'] for r in unknown_results.values())
    total_correct_unknown = sum(r['correct_unknown'] for r in unknown_results.values())
    overall_unknown_acc = total_correct_unknown / (total_unknown_samples + 1e-6)

    print(f"\n2. Unknown Class Metrics (Open-set Identification):")
    print(f"   - Total Unknown Samples Tested:   {total_unknown_samples}")
    print(f"   - Overall Unknown Rejection Rate: {overall_unknown_acc * 100:.2f}%")
    print("-" * 65)
    print("3. Fine-grained Unknown Misclassification Breakdown:")
    for u_cls, info in unknown_results.items():
        tot = info['total']
        corr = info['correct_unknown']
        rec_rate = (corr / tot * 100) if tot > 0 else 0.0
        print(f"   * Unknown [{u_cls}] (Total: {tot}):")
        print(f"     -> Correctly rejected as Unknown: {corr} ({rec_rate:.2f}%)")
        mis_list = [f"{cls_name}: {count}" for cls_name, count in info['misclassified_as'].items() if count > 0]
        if mis_list:
            print(f"     -> Misclassified Into: {', '.join(mis_list)}")
        else:
            print(f"     -> Misclassified Into: None (100% Detected)")
    print("="*65 + "\n")

    # (4) 绘制混淆矩阵
    plot_osr_confusion_matrix(known_true, known_pred, unknown_results, known_map, UNKNOWN_LABEL_ID)

# ==========================================
# 5. 混淆矩阵可视化展示
# ==========================================
def plot_osr_confusion_matrix(known_true, known_pred, unknown_results, known_map, UNKNOWN_ID, save_path="osr_confusion_matrix.png"):
    col_labels = [k for k, _ in sorted(known_map.items(), key=lambda item: item[1])] + ['Unknown']
    row_labels = col_labels[:-1] + sorted(list(unknown_results.keys()))

    cm = np.zeros((len(row_labels), len(col_labels)), dtype=np.int32)

    # 填充已知类预测情况
    for t, p in zip(known_true, known_pred):
        cm[t, p] += 1

    # 填充未知类详细预测情况
    for i, u_cls in enumerate(sorted(list(unknown_results.keys()))):
        row_idx = len(known_map) + i
        info = unknown_results[u_cls]
        cm[row_idx, UNKNOWN_ID] = info['correct_unknown']
        for k_name, count in info['misclassified_as'].items():
            k_id = known_map[k_name]
            cm[row_idx, k_id] = count

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=col_labels, yticklabels=row_labels)
    plt.title("Open Set Confusion Matrix (Knowns + Unknowns)", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Open set recognition confusion matrix saved to: {save_path}")
    plt.show()

# ==========================================
# 6. 主执行入口
# ==========================================
def main():
    TRAIN_DIR = "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms"  # 所有 .h5 文件的存储目录
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EPOCHS = 100
    PATIENCE = 10
    LR = 1e-4

    set_seed(42)

    KNOWN_CLASSES = ['A', 'D', 'E', 'G']
    UNKNOWN_CLASSES = ['B', 'C', 'F']

    # 1. 划分数据集 (文件级防泄露)
    tr_data, val_data, te_known, te_unknown, known_map = split_datasets_by_files(
        TRAIN_DIR, KNOWN_CLASSES, UNKNOWN_CLASSES
    )

    train_ds = STFTSplitDataset(tr_data[0], tr_data[1])
    val_ds   = STFTSplitDataset(val_data[0], val_data[1])
    test_k_ds = STFTSplitDataset(te_known[0], te_known[1])
    test_u_ds = STFTSplitDataset(te_unknown[0], te_unknown[1])

    train_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader    = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_k_loader = DataLoader(test_k_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_u_loader = DataLoader(test_u_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. 构建模型与度量学习头
    extractor = FullFeatureExtractor(win_size=32, stride=16, d_model=512).to(DEVICE)
    arcface_head = ArcMarginProduct(in_features=512, out_features=len(KNOWN_CLASSES), s=30.0, m=0.50).to(DEVICE)

    # 3. 训练模型 (10代早停)
    extractor, arcface_head = train_model(
        extractor, arcface_head, train_loader, val_loader, 
        device=DEVICE, epochs=EPOCHS, patience=PATIENCE, lr=LR
    )

    # 4. 基于验证集自适应确定拒绝阈值 (保证已知类 95% 接受度)
    threshold = calibrate_rejection_threshold(extractor, arcface_head, val_loader, device=DEVICE, target_recall=0.95)

    # 5. 开放集识别测试并生成全面指标报告
    evaluate_open_set(
        extractor, arcface_head, 
        test_k_loader, test_u_loader, 
        threshold=threshold, known_map=known_map, 
        unknown_classes=UNKNOWN_CLASSES, device=DEVICE
    )

if __name__ == "__main__":
    main()