import os
import glob
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# 0. 核心配置区 (集中管理，一键修改注册类别)
# ==========================================
CONFIG = {
    "data_dir": "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms",
    
    # ★★★【核心修改项】一键指定注册类别 (支持 'A', 'C', 'D', 'E', 'F', 'G') ★★★
    "target_class": "A",             # 注册目标类别：更改此项即可全流程自动适配
    
    "batch_size": 4,                 # 原生 [2048, 2055] 大图输入建议为 4
    "num_workers": 2,
    "lr": 1e-3,
    "epochs": 100,
    "early_stop_patience": 10,       # 10 轮无显著提升即早停
    "min_delta": 1e-3,               # 早停最小有效提升门限
    "embedding_dim": 64,             # 表征空间特征维度
    
    # 策略控制
    "include_b_in_train": False,     # True: B 参与训练; False: B 完全不参与训练，全作为未知非法测试
    "test_ratio": 0.20,              # 各类别切分测试集的百分比
    "val_ratio": 0.15,               # 训练数据中用于验证早停的比例
    "target_enroll_count": 100,      # 目标类别注册模板样本数 (上限 200)
    "target_recall_quantile": 0.95,  # 目标类别的理论召回门限分位数 (95%)
    "top_k_match": 3,                # 注册图库 Top-K 比对
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

# ==========================================
# 1. 原始双通道 IQ 矩阵读取类
# ==========================================
class STFTSplitDataset(Dataset):
    """
    原始双通道 IQ 时频矩阵读取类
    输出张量维度: [2, 2048, 2055]
    通道 0: Real (实部 I)
    通道 1: Imag (虚部 Q)
    """
    def __init__(self, file_paths, labels=None):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        h5_path = self.file_paths[idx]
        with h5py.File(h5_path, 'r') as f:
            real = f['real'][:].T
            imag = f['imag'][:].T
            tensor_data = np.stack([real, imag], axis=0).astype(np.float32)

        data = torch.from_numpy(tensor_data)
        if self.labels is not None:
            return data, self.labels[idx], os.path.basename(h5_path)
        return data, os.path.basename(h5_path)

# ==========================================
# 2. 网络结构与 CosFace 损失
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class SignalFeatureExtractor(nn.Module):
    """接收整张原生 [2, 2048, 2055] 输入，输出归一化特征"""
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3, bias=False), # -> 1024 x 1028
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                  # -> 512 x 514
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)   # -> 256 x 257
        self.layer2 = ResidualBlock(64, 128, stride=2)  # -> 128 x 129
        self.layer3 = ResidualBlock(128, 256, stride=2) # -> 64 x 65
        self.layer4 = ResidualBlock(256, 512, stride=2) # -> 32 x 33

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        feat = self.fc(x)
        return F.normalize(feat, p=2, dim=1)

class CosFaceHead(nn.Module):
    def __init__(self, embedding_dim=64, num_classes=5, scale=24.0, margin=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, feat_norm, labels):
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = torch.matmul(feat_norm, w_norm.T)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return self.scale * (cosine - one_hot * self.margin)

# ==========================================
# 3. 动态类别防数据泄露划分
# ==========================================
def parse_and_split_dataset(data_dir):
    target_cls = CONFIG["target_class"].upper()
    all_files = glob.glob(os.path.join(data_dir, "*.h5"))
    grouped = {}
    for f in all_files:
        label = os.path.basename(f)[0].upper()
        grouped.setdefault(label, []).append(f)

    for k in grouped:
        random.shuffle(grouped[k])

    # 基础候选类别池 (A, C, D, E, F, G)
    base_candidates = ['A', 'C', 'D', 'E', 'F', 'G']
    
    # 训练集类别自动剔除当前的目标类别
    train_classes = [c for c in base_candidates if c != target_cls]
    if CONFIG["include_b_in_train"] and target_cls != 'B':
        train_classes.append('B')
    train_classes.sort()

    cls_to_idx = {cls: idx for idx, cls in enumerate(train_classes)}

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    illegal_test_dict = {}

    test_ratio = CONFIG["test_ratio"]
    val_ratio = CONFIG["val_ratio"]

    # 1. 划分训练类别的样本 (按百分比切分防泄露)
    for cls in train_classes:
        if cls not in grouped:
            continue
        files = grouped[cls]
        n_total = len(files)
        n_test = max(1, int(n_total * test_ratio))
        n_remain = n_total - n_test
        n_val = max(1, int(n_remain * val_ratio))
        
        test_slice = files[:n_test]
        val_slice = files[n_test:n_test + n_val]
        train_slice = files[n_test + n_val:]

        illegal_test_dict[cls] = test_slice
        for f in train_slice:
            train_files.append(f)
            train_labels.append(cls_to_idx[cls])
        for f in val_slice:
            val_files.append(f)
            val_labels.append(cls_to_idx[cls])

    # 2. 如果 B 不参与训练且 B 不是目标类，全部划入非法测试集
    if not CONFIG["include_b_in_train"] and 'B' in grouped and target_cls != 'B':
        illegal_test_dict['B'] = grouped['B']

    # 3. 目标类别的划分 (完全独立，绝不参与模型训练)
    target_files = grouped.get(target_cls, [])
    if len(target_files) == 0:
        raise ValueError(f"数据目录中未找到类别 [{target_cls}] 的任何样本，请核查文件名前缀！")

    n_target_total = len(target_files)
    n_target_test = int(n_target_total * test_ratio)
    
    target_test_files = target_files[:n_target_test]
    target_enroll_pool = target_files[n_target_test:]

    enroll_cnt = min(CONFIG["target_enroll_count"], 200, len(target_enroll_pool))
    target_enroll_files = target_enroll_pool[:enroll_cnt]

    return (train_files, train_labels), (val_files, val_labels), target_enroll_files, target_test_files, illegal_test_dict, train_classes

# ==========================================
# 4. 模型训练与早停记录
# ==========================================
def train_feature_extractor(train_loader, val_loader, num_classes):
    device = CONFIG["device"]
    model = SignalFeatureExtractor(embedding_dim=CONFIG["embedding_dim"]).to(device)
    classifier = CosFaceHead(embedding_dim=CONFIG["embedding_dim"], num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=CONFIG["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_val_loss = float('inf')
    no_improve_epochs = 0
    patience = CONFIG["early_stop_patience"]
    min_delta = CONFIG["min_delta"]
    best_weights_path = "best_feature_extractor.pth"

    print("\n" + "=" * 65)
    print(f" 开始训练原生时频矩阵提取器 (类别数: {num_classes} | 早停: {patience} 轮 | 门限: {min_delta})")
    print("=" * 65)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                feats = model(x)
                logits = classifier(feats, y)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total

        # 验证
        model.eval()
        classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast():
                    feats = model(x)
                    logits = classifier(feats, y)
                    loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += len(y)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch:03d}/{CONFIG['epochs']:03d}] "
              f"Train Loss: {train_loss:.5f} | Acc: {train_acc:.2%} || "
              f"Val Loss: {val_loss:.5f} | Acc: {val_acc:.2%}")

        # 早停与权重跟踪
        improvement = best_val_loss - val_loss
        if improvement > min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"  └─> [★ 保存最佳权重] 改善: {improvement:.5f} (最优 Val Loss: {best_val_loss:.5f})")
            print(f"  └─> 早停计数器重置: [0/{patience}]")
        else:
            no_improve_epochs += 1
            print(f"  └─> [- 未达改善门限] 改善: {improvement:.5f} <= {min_delta} | 早停计数: [{no_improve_epochs}/{patience}]")

        print("-" * 65)

        if no_improve_epochs >= patience:
            print(f"### [触发早停终止]: 连续 {patience} 轮无有效改善，载入历史最优权重。")
            break

    model.load_state_dict(torch.load(best_weights_path))
    return model

# ==========================================
# 5. 特征提取与 Top-K 图库比对函数
# ==========================================
@torch.no_grad()
def extract_embeddings(model, file_list):
    device = CONFIG["device"]
    model.eval()
    dataset = STFTSplitDataset(file_list)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    all_feats = []
    for x, _ in loader:
        x = x.to(device)
        with autocast():
            feats = model(x)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)

def compute_gallery_similarity(query_feats, gallery_feats, top_k=3):
    sim_matrix = torch.mm(query_feats, gallery_feats.T)
    k = min(top_k, sim_matrix.size(1))
    topk_sims, _ = torch.topk(sim_matrix, k=k, dim=1)
    return torch.mean(topk_sims, dim=1).numpy()

# ==========================================
# 6. 主执行流程
# ==========================================
def main():
    target_cls = CONFIG["target_class"].upper()
    print("==================================================")
    print(f"   UAV 目标信号 [{target_cls}] 开放集表征比对辨识系统 (原生输入)   ")
    print("==================================================")

    train_data, val_data, target_enroll_files, target_test_files, illegal_test_dict, train_classes = parse_and_split_dataset(CONFIG["data_dir"])

    print(f"训练背景类别: {train_classes} (共 {len(train_classes)} 类)")
    print(f"训练样本总数: {len(train_data[0])} | 验证样本数: {len(val_data[0])}")
    print(f"目标信号 [{target_cls}] 注册样本数: {len(target_enroll_files)} (上限 200)")
    print(f"目标信号 [{target_cls}] 测试样本数: {len(target_test_files)} (严格独立于注册池)")
    print("各非法类别预留测试集 (完全未见):")
    for cls, f_list in illegal_test_dict.items():
        print(f"  -> 非法类别 [{cls}]: {len(f_list)} 个盲测样本")

    train_loader = DataLoader(STFTSplitDataset(*train_data), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(STFTSplitDataset(*val_data), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 1. 训练表征提取器
    model = train_feature_extractor(train_loader, val_loader, num_classes=len(train_classes))

    # 2. 构建目标类别的注册图库 (Gallery)
    print(f"\n>>> 正在提取并构建目标类别 [{target_cls}] 的特征注册图库 (Gallery Matrix)...")
    gallery_feats = extract_embeddings(model, target_enroll_files)

    # 3. 留一法 (Leave-One-Out) 自适应标定接受阈值
    internal_sims = []
    for i in range(len(gallery_feats)):
        feat = gallery_feats[i:i+1]
        other_feats = torch.cat([gallery_feats[:i], gallery_feats[i+1:]], dim=0)
        score = compute_gallery_similarity(feat, other_feats, top_k=CONFIG["top_k_match"])[0]
        internal_sims.append(score)
    internal_sims = np.array(internal_sims)

    calibrated_threshold = float(np.quantile(internal_sims, 1.0 - CONFIG["target_recall_quantile"]))
    print(f"\n[阈值自适应标定 (基于 Top-{CONFIG['top_k_match']} 图库匹配)]:")
    print(f"  -> [{target_cls}] 注册集内部互验均值: {np.mean(internal_sims):.4f}, 最低={np.min(internal_sims):.4f}")
    print(f"  -> 保证 {CONFIG['target_recall_quantile']*100:.1f}% 召回的判决门限: 【{calibrated_threshold:.4f}】")

    # 4. 执行开集测试打分
    test_scores = {}
    
    # 目标类别测试
    target_test_feats = extract_embeddings(model, target_test_files)
    test_scores[target_cls] = compute_gallery_similarity(target_test_feats, gallery_feats, top_k=CONFIG["top_k_match"])

    # 其余非法类别测试
    for cls, files in illegal_test_dict.items():
        feats = extract_embeddings(model, files)
        test_scores[cls] = compute_gallery_similarity(feats, gallery_feats, top_k=CONFIG["top_k_match"])

    # 5. 打印得分分布全景表
    print("\n" + "=" * 65)
    print(f"       各类别测试集与 [{target_cls}]-Gallery(Top-{CONFIG['top_k_match']}) 相似度分布全景        ")
    print("=" * 65)
    print(f"{'类别':<8}{'样本量':<10}{'最小值(Min)':<14}{'均值(Mean)':<14}{'最大值(Max)':<14}{'标准差(Std)':<10}")
    print("-" * 65)
    all_eval_classes = [target_cls] + sorted([k for k in test_scores.keys() if k != target_cls])
    for cls in all_eval_classes:
        scores = test_scores[cls]
        print(f"{cls:<8}{len(scores):<10}{np.min(scores):<14.4f}{np.mean(scores):<14.4f}{np.max(scores):<14.4f}{np.std(scores):<10.4f}")
    print("=" * 65)

    # 6. 最终辨识报告
    print("\n" + "=" * 65)
    print("                 开集测试最终辨识报告                  ")
    print(f"判定准则: 相似度 >= {calibrated_threshold:.4f} 判定为 [{target_cls}], 否则为非法信号")
    print("=" * 65)

    target_pred_correct = np.sum(test_scores[target_cls] >= calibrated_threshold)
    total_test_target = len(test_scores[target_cls])
    target_recall = (target_pred_correct / total_test_target) * 100 if total_test_target > 0 else 0
    print(f"【目标信号 [{target_cls}] 识别结果】:")
    print(f"  -> 正确预测为 [{target_cls}] 的样本数: {target_pred_correct} / {total_test_target}")
    print(f"  -> 占 [{target_cls}] 全部输入测试样本的比例 (召回率): {target_recall:.2f}%")
    print("-" * 65)

    total_illegal_tested = 0
    total_illegal_rejected = 0
    print(f"【非法信号拦截结果 (判定为非 [{target_cls}])】:")
    for cls in all_eval_classes:
        if cls == target_cls:
            continue
        scores = test_scores[cls]
        rejected = np.sum(scores < calibrated_threshold)
        total_illegal_tested += len(scores)
        total_illegal_rejected += rejected
        rej_rate = (rejected / len(scores)) * 100 if len(scores) > 0 else 0
        print(f"  -> 类别 [{cls}]: 正确识别为非法信号 {rejected}/{len(scores)} (拦截率: {rej_rate:.2f}%)")

    overall_rej = (total_illegal_rejected / total_illegal_tested) * 100 if total_illegal_tested > 0 else 0
    print(f"\n  -> 全部非法信号总拦截数: {total_illegal_rejected} / {total_illegal_tested}")
    print(f"  -> 非法信号综合拦截准确率: {overall_rej:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()