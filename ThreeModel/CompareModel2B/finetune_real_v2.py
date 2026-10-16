"""
AMR 混合域安全微调脚本 v2 (支持跨设备数据对齐与骨干网络硬冻结)
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
# ║  1. CONFIGURATION — 安全微调配置                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
FT_CONFIG = {
    'model_name': 'tcn',
    'pretrain_weights': 'weights/tcn_v2_SionnaA_6.pt',
    'sim_dataset_path': './sionna_dataset/SionnaA_6.npz',
    'real_dataset_root': 'E:/Project/AMR-Benchmark/real_dataset/',
    'save_weights_path': 'weights/tcn_v2_finetuned2_real2.pt',
    'save_confusion_path': 'figure/tcn_v2_finetune_confusion.png',

    # ---- 安全微调核心控制 ----
    'freeze_backbone': True,  # 强烈推荐设为 True！完全冻结骨干特征提取，100% 保护 256QAM 特征不漂移
    'sim_anchor_ratio': 1.0,  # 有实采数据的类，仿真锚点与实采样本 1:1 混合
    'missing_class_anchor': 800,  # 缺失实采数据的类（如256QAM），在训练集中强制灌入 800 个仿真锚点以撑开决策空间

    # ---- 训练超参数 ----
    'epochs': 40,
    'batch_size': 128,
    'backbone_lr': 1e-6,  # 若 freeze_backbone 为 False，此极小学习率生效
    'head_lr': 5e-4,  # 分类输出头的微调学习率
}

MODEL_NAME = FT_CONFIG['model_name']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Mode: Secure Fine-tuning [{MODEL_NAME}]")

if MODEL_NAME == 'transformer':
    from model_transformer_v2 import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn_v2 import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. 混合域数据加载与平衡器                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
def load_real_data(root_dir, classes_list):
    """自动扫描实采文件夹，并支持跨设备数据导入"""
    real_x_list = []
    real_y_list = []

    print("\n[实采数据扫描] 正在读取本地文件夹...")
    for cls_idx, cls_name in enumerate(classes_list):
        pattern = os.path.join(root_dir, f"*{cls_name}*")
        matched_dirs = glob.glob(pattern)

        if not matched_dirs:
            continue

        target_dir = matched_dirs[0]
        mat_files = glob.glob(os.path.join(target_dir, "*.mat"))
        if len(mat_files) == 0:
            continue

        print(f"  ✓ 识别到 [{cls_name}] 实采数据: {os.path.basename(target_dir)} ({len(mat_files)} 样本)")

        for fpath in mat_files:
            try:
                with h5py.File(fpath, 'r') as f:
                    key = 'data_save' if 'data_save' in f else list(f.keys())[0]
                    arr = f[key][()]
                    arr = np.squeeze(arr)

                    if arr.shape == (2, 1024):
                        real_x_list.append(arr.astype(np.float32))
                        real_y_list.append(cls_idx)
            except Exception:
                pass

    if len(real_x_list) == 0:
        return None, None

    X_real = np.stack(real_x_list, axis=0)
    Y_real = np.array(real_y_list, dtype=np.int64)
    return X_real, Y_real


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. 损失函数与标准 IQ 数据集定义                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.02, weight=None):
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
        # 微调轻度增强
        if np.random.random() < 0.2:
            shift = np.random.randint(-10, 10)
            sample = np.roll(sample, shift, axis=1)
        if np.random.random() < 0.2:
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
        sample_tensor = sample_tensor.permute(1, 0)
        return sample_tensor, self.labels[idx]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. 微调控制核心逻辑                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
def main():
    print("Loading Base Simulation Dataset...")
    sim_data = np.load(FT_CONFIG['sim_dataset_path'], allow_pickle=True)
    classes = list(sim_data['classes'])
    print(f"Dataset classes: {classes}")

    test_raw = sim_data['X_test']
    test_labels_idx = sim_data['Y_test']
    Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
    Z_test = sim_data['snr_test'].reshape(-1, 1)

    # 加载实采微调数据 (自动包含跨设备采集的 256QAM 等数据)
    X_real, Y_real = load_real_data(FT_CONFIG['real_dataset_root'], classes)

    # 动态构建平衡数据集，防止决策边界坍缩
    sim_train_raw = sim_data['X_train']
    sim_train_labels = sim_data['Y_train']

    anchor_x_list = []
    anchor_y_list = []

    print("\n[平衡数据集构建] 正在生成混合域训练集...")
    for cls_idx, cls_name in enumerate(classes):
        cls_real_mask = (Y_real == cls_idx) if Y_real is not None else np.array([])
        n_real = cls_real_mask.sum()

        cls_sim_indices = np.where(sim_train_labels == cls_idx)[0]
        high_snr_indices = cls_sim_indices[len(cls_sim_indices) // 2:]  # 取仿真中高SNR数据

        if n_real > 0:
            # 1. 存在实采数据的类别：进行 1:1 混合
            real_x = X_real[cls_real_mask]
            real_y = Y_real[cls_real_mask]

            n_anchors = int(n_real * FT_CONFIG['sim_anchor_ratio'])
            sampled = np.random.choice(high_snr_indices, size=min(n_anchors, len(high_snr_indices)), replace=False)

            anchor_x_list.append(sim_train_raw[sampled])
            anchor_y_list.append(sim_train_labels[sampled])
            anchor_x_list.append(real_x)
            anchor_y_list.append(real_y)
            print(f"  -> 类别 {cls_name:8s}: 混合训练 (实采={n_real} 样本, 仿真锚点={len(sampled)} 样本)")
        else:
            # 2. 完全缺失实采数据的类别（如 256QAM）：强制注入大比例仿真锚点，撑开其判决空间
            n_anchors = FT_CONFIG['missing_class_anchor']
            sampled = np.random.choice(high_snr_indices, size=min(n_anchors, len(high_snr_indices)), replace=False)

            anchor_x_list.append(sim_train_raw[sampled])
            anchor_y_list.append(sim_train_labels[sampled])
            print(f"  -> 类别 {cls_name:8s}: 纯仿真锚点填充 (仿真锚点={len(sampled)} 样本) [保护该类不被挤压]")

    X_ft_train = np.concatenate(anchor_x_list, axis=0)
    Y_ft_train = np.concatenate(anchor_y_list, axis=0)

    # 划分微调验证集 (10%)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_ft_train, Y_ft_train, test_size=0.1, random_state=42, stratify=Y_ft_train
    )

    print(f"\n数据集汇总: 训练集规模={len(X_train)} | 验证集规模={len(X_val)}")

    train_dataset = IQDataset(X_train, Y_train, normalize=True, augment=True)
    val_dataset = IQDataset(X_val, Y_val, normalize=True, augment=False)
    test_dataset = IQDataset(test_raw, test_labels_idx, normalize=True, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=True, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=False, num_workers=0,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=FT_CONFIG['batch_size'], shuffle=False, num_workers=0,
                             pin_memory=True)

    # 加载预训练模型
    model = build_model(classes=len(classes)).to(device)
    print(f"\nLoading weights from: {FT_CONFIG['pretrain_weights']}")
    model.load_state_dict(torch.load(FT_CONFIG['pretrain_weights'], map_location=device))

    # ---- 骨干特征提取器硬冻结控制 ----
    if FT_CONFIG['freeze_backbone']:
        print("  [硬冻结激活] 骨干特征网络已锁定，仅训练最后一层 FC 分类输出头 (100% 保护 256QAM 特征)")
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name or 'linear' in name or 'out' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=FT_CONFIG['head_lr'], weight_decay=1e-4)
    else:
        print("  [硬冻结未激活] 骨干网络以极低学习率参与调整")
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name or 'linear' in name or 'out' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': FT_CONFIG['backbone_lr']},
            {'params': head_params, 'lr': FT_CONFIG['head_lr']}
        ], weight_decay=1e-4)

    # 均衡 QAM 权重
    class_weights = torch.ones(len(classes), dtype=torch.float32, device=device)
    class_weights[classes.index('128QAM')] = 1.5
    if '256QAM' in classes:
        class_weights[classes.index('256QAM')] = 1.3

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.02, weight=class_weights)
    nb_epoch = FT_CONFIG['epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epoch, eta_min=1e-7)
    scaler = GradScaler()

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(nb_epoch):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epoch}")

        for inputs, labels in train_pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix(
                {'loss': f'{train_loss / train_total:.4f}', 'acc': f'{train_correct / train_total:.4f}'})

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
            print(f"  ✓ Saved best weights. ({best_val_acc:.4f} → {avg_val_acc:.4f})")
            torch.save(model.state_dict(), FT_CONFIG['save_weights_path'])
            best_val_acc = avg_val_acc

    # 测试集评估
    print("\n--- Final Evaluation ---")
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
    print(f'\n=== Fine-tuned Model Accuracy: {acc * 100:.2f}% ===')
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename=FT_CONFIG['save_confusion_path'])

    # 逐 QAM 类型细致统计
    print(f"\n{'=' * 40}")
    print("QAM Per-Class Accuracy Evaluation:")
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