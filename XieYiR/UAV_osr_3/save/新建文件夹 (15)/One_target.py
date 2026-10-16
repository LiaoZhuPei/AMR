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
# 0. 核心配置区 (集中管理，无 YAML)
# ==========================================
CONFIG = {
    "data_dir": "E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train_h5_spectrograms",
    "batch_size": 4,                 # 原生矩阵 [2048, 2055] 整图输入
    "num_workers": 2,
    "lr": 1e-3,
    "epochs": 100,
    "early_stop_patience": 10,       # 早停忍耐轮数
    "min_delta": 1e-3,               # 早停最小有效改善门限
    "embedding_dim": 64,             # 表征空间特征维度
    
    # 开放集比对策略
    "target_class": "D",             # 【核心设置】: 可任意指定目标类别 ('A'~'G')
    "test_ratio": 0.20,              # 各类别硬性切分测试集的百分比 (样本级隔离)
    "val_ratio": 0.15,               # 训练数据中用于早停验证的比例
    "target_enroll_count": 100,      # 目标信号注册模板数量 (上限 200)
    "target_recall_quantile": 0.95,  # 目标信号自适应召回保障线 (95%)
    "top_k_match": 3,                # 注册图库 Top-K 细粒度比对
    
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
# 1. 原始双通道 IQ 矩阵读取类 (完全不做人为变换)
# ==========================================
class STFTSplitDataset(Dataset):
    """
    原生输入: [2, 2048, 2055]
    通道 0: 实部 Real (I)
    通道 1: 虚部 Imag (Q)
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
# 2. 网络结构与 CosFace 度量损失
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
    def __init__(self, embedding_dim=64):
        super().__init__()
        # Stem 模块: 第一时间下采样，杜绝超大矩阵显存溢出
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
        return F.normalize(feat, p=2, dim=1) # 投影至超球面

class CosFaceHead(nn.Module):
    def __init__(self, embedding_dim=64, num_classes=7, scale=24.0, margin=0.25):
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
# 3. 全局数据解析与样本级防泄露划分
# ==========================================
def parse_and_split_dataset(data_dir, target_class):
    all_files = glob.glob(os.path.join(data_dir, "*.h5"))
    grouped = {}
    for f in all_files:
        label = os.path.basename(f)[0].upper()
        grouped.setdefault(label, []).append(f)

    all_classes = sorted(list(grouped.keys()))
    assert target_class in all_classes, f"目标类别 {target_class} 不在数据集中: {all_classes}"
    
    cls_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    all_test_dict = {}

    test_ratio = CONFIG["test_ratio"]
    val_ratio = CONFIG["val_ratio"]

    # 遍历所有类别：按百分比严格划分，杜绝任何样本重叠
    for cls in all_classes:
        files = grouped[cls]
        random.shuffle(files)
        
        n_total = len(files)
        n_test = max(1, int(n_total * test_ratio)) # 保底 1 个盲测样本
        n_remain = n_total - n_test
        n_val = max(1, int(n_remain * val_ratio))
        
        test_slice = files[:n_test]
        val_slice = files[n_test:n_test + n_val]
        train_slice = files[n_test + n_val:]

        all_test_dict[cls] = test_slice

        for f in train_slice:
            train_files.append(f)
            train_labels.append(cls_to_idx[cls])
        for f in val_slice:
            val_files.append(f)
            val_labels.append(cls_to_idx[cls])

    # 针对目标类别，从其训练池中选取注册模板 (代表授权入库样本)
    target_train_candidates = [f for f, l in zip(train_files, train_labels) if l == cls_to_idx[target_class]]
    enroll_cnt = min(CONFIG["target_enroll_count"], 200, len(target_train_candidates))
    target_enroll_files = target_train_candidates[:enroll_cnt]
    
    # 目标测试样本来自其纯盲测集 (模型训练过程中从未见过)
    target_test_files = all_test_dict[target_class]
    
    # 非法测试集包含除目标类别外的所有类别的纯盲测集
    illegal_test_dict = {cls: files for cls, files in all_test_dict.items() if cls != target_class}

    return (train_files, train_labels), (val_files, val_labels), target_enroll_files, target_test_files, illegal_test_dict, all_classes

# ==========================================
# 4. 全局模型训练与早停
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
    print(f" 开始训练全类别表征提取器 (总类别数: {num_classes} | 早停: {patience} 轮 | 门限: {min_delta})")
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

        # 早停与权重保存
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
# 5. 特征提取与 Top-K 比对
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
    target_cls = CONFIG["target_class"]
    print("==================================================")
    print(f"   UAV 目标信号 [{target_cls}] 开放集表征比对辨识系统 (全类训练)")
    print("==================================================")

    train_data, val_data, target_enroll_files, target_test_files, illegal_test_dict, all_classes = parse_and_split_dataset(
        CONFIG["data_dir"], target_class=target_cls
    )

    print(f"全部参与训练类别: {all_classes} (共 {len(all_classes)} 类)")
    print(f"训练集总样本数: {len(train_data[0])} | 验证集样本数: {len(val_data[0])}")
    print(f"当前指定目标信号: [{target_cls}]")
    print(f"  -> 授权注册样本数: {len(target_enroll_files)} (上限 200)")
    print(f"  -> 盲测测试样本数: {len(target_test_files)} (严格隔离自训练与早停)")
    print("其余类别预留非法测试集 (盲测样本):")
    for cls, f_list in illegal_test_dict.items():
        print(f"  -> 非法类别 [{cls}]: {len(f_list)} 个测试样本")

    train_loader = DataLoader(STFTSplitDataset(*train_data), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(STFTSplitDataset(*val_data), batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # 1. 训练表征提取器 (全部 7 类共同撑开特征空间)
    model = train_feature_extractor(train_loader, val_loader, num_classes=len(all_classes))

    # 2. 构建目标类别的注册图库 (Gallery)
    print(f"\n>>> 正在提取并构建目标信号 [{target_cls}] 的注册图库 (Gallery Matrix)...")
    target_gallery_feats = extract_embeddings(model, target_enroll_files)

    # 3. 留一法 (Leave-One-Out) 自适应标定目标信号接受阈值
    internal_sims = []
    for i in range(len(target_gallery_feats)):
        feat = target_gallery_feats[i:i+1]
        other_feats = torch.cat([target_gallery_feats[:i], target_gallery_feats[i+1:]], dim=0)
        score = compute_gallery_similarity(feat, other_feats, top_k=CONFIG["top_k_match"])[0]
        internal_sims.append(score)
    internal_sims = np.array(internal_sims)

    calibrated_threshold = float(np.quantile(internal_sims, 1.0 - CONFIG["target_recall_quantile"]))
    print(f"\n[阈值自适应标定 (基于 Top-{CONFIG['top_k_match']} 图库匹配)]:")
    print(f"  -> [{target_cls}] 注册集内部互验均值: {np.mean(internal_sims):.4f}, 最低={np.min(internal_sims):.4f}")
    print(f"  -> 保证 {CONFIG['target_recall_quantile']*100:.1f}% 召回的判决门限: 【{calibrated_threshold:.4f}】")

    # 4. 执行测试集特征提取与打分
    test_scores = {}
    
    # 目标信号盲测试
    target_test_feats = extract_embeddings(model, target_test_files)
    test_scores[target_cls] = compute_gallery_similarity(target_test_feats, target_gallery_feats, top_k=CONFIG["top_k_match"])

    # 各非法类别盲测试
    for cls, files in illegal_test_dict.items():
        feats = extract_embeddings(model, files)
        test_scores[cls] = compute_gallery_similarity(feats, target_gallery_feats, top_k=CONFIG["top_k_match"])

    # 5. 打印得分分布全景表
    print("\n" + "=" * 65)
    print(f"       各类别测试集与 [{target_cls}]-Gallery 相似度分布全景        ")
    print("=" * 65)
    print(f"{'类别':<8}{'样本量':<10}{'最小值(Min)':<14}{'均值(Mean)':<14}{'最大值(Max)':<14}{'标准差(Std)':<10}")
    print("-" * 65)
    eval_classes_order = [target_cls] + sorted([k for k in test_scores.keys() if k != target_cls])
    for cls in eval_classes_order:
        scores = test_scores[cls]
        print(f"{cls:<8}{len(scores):<10}{np.min(scores):<14.4f}{np.mean(scores):<14.4f}{np.max(scores):<14.4f}{np.std(scores):<10.4f}")
    print("=" * 65)

    # 6. 最终辨识报告
    print("\n" + "=" * 65)
    print("                 开集测试最终辨识报告                  ")
    print(f"判定准则: 相似度 >= {calibrated_threshold:.4f} 判定为 [{target_cls}]，否则为非法信号")
    print("=" * 65)

    target_pred_correct = np.sum(test_scores[target_cls] >= calibrated_threshold)
    total_test_target = len(test_scores[target_cls])
    target_recall = (target_pred_correct / total_test_target) * 100
    print(f"【目标信号 [{target_cls}] 识别结果】:")
    print(f"  -> 正确预测为 [{target_cls}] 的样本数: {target_pred_correct} / {total_test_target}")
    print(f"  -> 占 [{target_cls}] 全部盲测样本的比例 (召回率): {target_recall:.2f}%")
    print("-" * 65)

    total_illegal_tested = 0
    total_illegal_rejected = 0
    print(f"【非法信号拦截结果 (判定为非 {target_cls})】:")
    for cls in eval_classes_order:
        if cls == target_cls:
            continue
        scores = test_scores[cls]
        rejected = np.sum(scores < calibrated_threshold)
        total_illegal_tested += len(scores)
        total_illegal_rejected += rejected
        rej_rate = (rejected / len(scores)) * 100 if len(scores) > 0 else 0
        print(f"  -> 类别 [{cls}]: 正确识别为非法信号 {rejected}/{len(scores)} (拦截率: {rej_rate:.2f}%)")

    overall_rej = (total_illegal_rejected / total_illegal_tested) * 100
    print(f"\n  -> 全部非法信号总拦截数: {total_illegal_rejected} / {total_illegal_tested}")
    print(f"  -> 非法信号综合拦截准确率: {overall_rej:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()