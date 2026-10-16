"""
AMR 仿真-实采混合域迭代微调脚本 (finetune.py)
"""

import os
import time
import glob
import h5py
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

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION — 微调配置                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
FT_CONFIG = {
    'model_name': 'tcn',  # 'transformer', 'tcn'
    'pretrain_weights': 'weights/tcn_v2_SionnaA_6.pt',  # 预训练权重路径
    'sim_dataset_path': './sionna_dataset/newdata_1.npz',
    'real_dataset_root': 'E:/Project/AMR-Benchmark/real_dataset/',  # 实采数据根目录
    'save_weights_path': 'weights/tcn_v2_finetuned_real2.pt',
    'save_confusion_path': 'figure/tcn_v2_finetune_confusion.png',

    # ---- 训练超参数 ----
    'epochs': 50,  # 微调不需要1000轮，通常30-50轮即可收敛
    'batch_size': 128,  # 较小的 batch_size 适合小样本微调
    'backbone_lr': 1e-5,  # 骨干网络极小学习率，防止遗忘
    'head_lr': 1e-3,  # 分类头正常学习率，使其快速对齐新特征
    'sim_anchor_ratio': 1.0,  # 仿真锚点与实采样本的配比（1.0 表示 1:1 混合）
}

MODEL_NAME = FT_CONFIG['model_name']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Task: Fine-tuning [{MODEL_NAME}]")

if MODEL_NAME == 'transformer':
    from model_transformer_v2 import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn_v2 import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. 实采数据扫描与加载器                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
def load_real_data(root_dir, classes_list):
    """
    自动扫描实采文件夹，加载所有 .mat 文件并提取 IQ 数据
    返回: X_real (M, 2, 1024), Y_real (M,)
    """
    real_x_list = []
    real_y_list = []

    print("\n[实采数据扫描] 正在读取本地文件夹...")
    for cls_idx, cls_name in enumerate(classes_list):
        # 寻找匹配类名的文件夹，例如：E:/Project/AMR-Benchmark/real_dataset/*128QAM*
        pattern = os.path.join(root_dir, f"*{cls_name}*")
        matched_dirs = glob.glob(pattern)

        if not matched_dirs:
            continue

        target_dir = matched_dirs[0]
        mat_files = glob.glob(os.path.join(target_dir, "*.mat"))
        if len(mat_files) == 0:
            continue

        print(f"  识别到类 [{cls_name}] 的实采数据文件夹: {os.path.basename(target_dir)}，含有 {len(mat_files)} 个样本")

        for fpath in mat_files:
            try:
                with h5py.File(fpath, 'r') as f:
                    # 兼容 data_save 键值读取
                    key = 'data_save' if 'data_save' in f else list(f.keys())[0]
                    arr = f[key][()]  # 形状: (1, 2, 1024)
                    arr = np.squeeze(arr)  # 挤压成 (2, 1024)

                    if arr.shape == (2, 1024):
                        real_x_list.append(arr.astype(np.float32))
                        real_y_list.append(cls_idx)
            except Exception as e:
                pass

    if len(real_x_list) == 0:
        print("⚠ 警告: 未能在指定的根目录下扫描到任何合法的实采 .mat 数据！")
        return None, None

    X_real = np.stack(real_x_list, axis=0)
    Y_real = np.array(real_y_list, dtype=np.int64)
    print(
        f"[实采加载完成] 共加载实采样本数: {len(Y_real):,}, 覆盖类别数: {len(np.unique(Y_real))}/{len(classes_list)}\n")
    return X_real, Y_real


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. 损失函数与数据集定义                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=self.ls, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def power_normalize(signal_data):
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


class IQDataset(Dataset):
    def __init__(self, iq_data, labels, normalize=True, augment=False):
        self.raw_data = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.normalize = normalize
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, sample):
        # 针对微调实施轻度增强，避免破坏实采特征
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 10)
            sample = np.roll(sample, shift, axis=1)
        if np.random.random() < 0.3:
            theta = np.random.uniform(0, 2 * np.pi)
            I, Q = sample[0].copy(), sample[1].copy()
            sample[0] = I * np.cos(theta) - Q * np.sin(theta)
            sample[1] = I * np.sin(theta) + Q * np.cos(theta)
        return sample

    def __getitem__(self, idx):
        sample = self.raw_data[idx].copy()
        if self.augment:
            sample = self._augment(sample)
        if self.normalize:
            sample = power_normalize(sample)
        sample_tensor = torch.from_numpy(sample).float()
        sample_tensor = sample_tensor.permute(1, 0)  # (2, 1024) -> (1024, 2)
        return sample_tensor, self.labels[idx]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. 主训练控制逻辑                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
def main():
    # 1. 读取基础仿真数据和类别
    print("Loading Base Simulation Dataset...")
    sim_data = np.load(FT_CONFIG['sim_dataset_path'], allow_pickle=True)
    classes = list(sim_data['classes'])
    print(f"Dataset classes: {classes}")

    # 读取测试集 (作为微调完后的联合测试)
    test_raw = sim_data['X_test']
    test_labels_idx = sim_data['Y_test']
    Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
    Z_test = sim_data['snr_test'].reshape(-1, 1)

    # 2. 扫描并加载本地实采微调数据
    X_real, Y_real = load_real_data(FT_CONFIG['real_dataset_root'], classes)
    if X_real is None:
        print("错误: 未检测到任何实采微调数据集，退出。")
        return

    # 3. 构造混合域协同微调训练集 (防止遗忘)
    print("Constructing Joint Sim-Real Dataset for Fine-Tuning...")
    # 提取仿真训练集中高 SNR 状态下的优质样本作为长期记忆锚点
    sim_train_raw = sim_data['X_train']
    sim_train_labels = sim_data['Y_train']

    # 针对每个类别抽样仿真锚点数据，以匹配实采规模
    n_anchors_per_class = int(len(X_real) / len(classes) * FT_CONFIG['sim_anchor_ratio'])
    n_anchors_per_class = max(100, n_anchors_per_class)  # 确保每类至少有100个仿真锚点

    anchor_x_list = []
    anchor_y_list = []
    for cls_idx in range(len(classes)):
        cls_indices = np.where(sim_train_labels == cls_idx)[0]
        # 优先抽取仿真数据集里后半段高信噪比区域的数据
        high_snr_indices = cls_indices[len(cls_indices) // 2:]
        sampled = np.random.choice(high_snr_indices, size=min(n_anchors_per_class, len(high_snr_indices)),
                                   replace=False)
        anchor_x_list.append(sim_train_raw[sampled])
        anchor_y_list.append(sim_train_labels[sampled])

    X_sim_anchors = np.concatenate(anchor_x_list, axis=0)
    Y_sim_anchors = np.concatenate(anchor_y_list, axis=0)

    # 将仿真锚点与实采数据完全合并
    X_ft_train = np.concatenate([X_sim_anchors, X_real], axis=0)
    Y_ft_train = np.concatenate([Y_sim_anchors, Y_real], axis=0)

    # 划分微调验证集 (包含10%混合域数据)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_ft_train, Y_ft_train, test_size=0.1, random_state=42, stratify=Y_ft_train
    )

    print(f"Fine-tuning Dataset Summary:")
    print(
        f"  -> Total Combined Train Size: {len(X_train)} (Sim anchors: {len(X_sim_anchors)}, Real captured: {len(X_real)})")
    print(f"  -> Total Validation Size:     {len(X_val)}")
    print(f"  -> Evaluated on base test:    {len(test_raw)}")

    # 4. 创建 Dataloader
    train_dataset = IQDataset(X_train, Y_train, normalize=True, augment=True)
    val_dataset = IQDataset(X_val, Y_val, normalize=True, augment=False)
    test_dataset = IQDataset(test_raw, test_labels_idx, normalize=True, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=True, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=False, num_workers=0,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=False, num_workers=0,
                             pin_memory=True)

    # 5. 加载预训练模型与差分学习率设定
    model = build_model(classes=len(classes)).to(device)
    print(f"\nLoading Pre-trained weights from: {FT_CONFIG['pretrain_weights']}")
    model.load_state_dict(torch.load(FT_CONFIG['pretrain_weights'], map_location=device))

    # --- 差分学习率分组 (Differential Learning Rate) ---
    # 分类头模块通常含有 "fc", "classifier" 或 "linear" 命名，骨干网络则为除此以外的所有模块
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'linear' in name or 'out' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    # 若名称不匹配，执行兜底分割
    if len(head_params) == 0:
        all_params = list(model.parameters())
        backbone_params = all_params[:-2]
        head_params = all_params[-2:]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': FT_CONFIG['backbone_lr']},  # 骨干网络极小LR，保持对仿真特征的记忆
        {'params': head_params, 'lr': FT_CONFIG['head_lr']}  # 分类输出头正常LR，快速对齐实采特征
    ], weight_decay=1e-4)

    # 6. 微调损失函数与训练周期控制
    # 增加微调类别权重均衡
    class_weights = torch.ones(len(classes), dtype=torch.float32, device=device)
    class_weights[classes.index('128QAM')] = 1.8  # 特别对 128QAM 实采极重灾区加大权重
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.02, weight=class_weights)

    nb_epoch = FT_CONFIG['epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epoch, eta_min=1e-7)
    scaler = GradScaler()

    # 7. 微调主循环
    best_val_acc = 0.0
    for epoch in range(nb_epoch):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch + 1}/{nb_epoch}")

        for inputs, labels in train_pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                'loss': f'{train_loss / train_total:.4f}',
                'acc': f'{train_correct / train_total:.4f}',
            })

        scheduler.step()

        # 验证评估
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_acc = val_correct / val_total
        print(f"  Epoch {epoch + 1}: TrainAcc={train_correct / train_total:.4f} ValAcc={avg_val_acc:.4f}")

        if avg_val_acc > best_val_acc:
            print(f"  ✓ Saved fine-tuned weights. ({best_val_acc:.4f} → {avg_val_acc:.4f})")
            torch.save(model.state_dict(), FT_CONFIG['save_weights_path'])
            best_val_acc = avg_val_acc

    # 8. 全面测试集评估
    print("\n--- Running Final Evaluation ---")
    model.load_state_dict(torch.load(FT_CONFIG['save_weights_path'], map_location=device))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="[Testing]"):
            inputs = inputs.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
            probs = torch.softmax(outputs.float(), dim=1)
            all_preds.append(probs.cpu().numpy())

    test_Y_hat = np.concatenate(all_preds, axis=0)

    # 绘制混淆矩阵
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'\n=== Fine-tuned Model Overall Accuracy: {acc * 100:.2f}% ===')
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename=FT_CONFIG['save_confusion_path'])

    # 打印各类测试集的最终准确度
    print(f"\n{'=' * 40}")
    print("QAM Per-Class Evaluation after Fine-tuning:")
    y_true = np.argmax(Y_test_onehot, axis=1)
    y_pred = np.argmax(test_Y_hat, axis=1)
    qam_classes = ['16QAM', '32QAM', '64QAM', '128QAM', '256QAM']
    for cls_name in qam_classes:
        if cls_name in classes:
            cls_idx = classes.index(cls_name)
            mask = (y_true == cls_idx)
            if mask.sum() > 0:
                cls_acc = (y_pred[mask] == cls_idx).sum() / mask.sum()
                print(f"  {cls_name:>8s}: {cls_acc * 100:.2f}%  ({mask.sum()} samples)")
    print(f"{'=' * 40}")


if __name__ == '__main__':
    main()