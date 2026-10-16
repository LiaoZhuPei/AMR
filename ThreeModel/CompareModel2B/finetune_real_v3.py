"""
AMR 高稳定度微调脚本 (stable_finetune.py)
引入 LwF 知识蒸馏与 L2-SP 参数约束，彻底解决灾难性遗忘问题
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
# ║  1. CONFIGURATION — 稳定微调配置                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
FT_CONFIG = {
    'model_name': 'tcn',
    'pretrain_weights': 'weights/tcn_v2_newdata_1.pt',
    'sim_dataset_path': './sionna_dataset/newdata_1.npz',
    'real_dataset_root': 'E:/Project/AMR-Benchmark/real_dataset/',
    'save_weights_path': 'weights/tcn_v2_stable_finetuned.pt',
    'save_confusion_path': 'figure/tcn_v2_stable_finetune_confusion.png',

    # ---- 稳定性核心超参数 ----
    'kd_alpha': 0.6,  # 知识保留权重 (0.0~1.0)。值越大，对 A, B, C 的保护越强
    'kd_temperature': 2.0,  # 蒸馏温度。用于平滑 Softmax 概率，保留类间细微的相似度关系
    'l2_sp_lambda': 1e-4,  # 权重漂移惩罚系数。限制分类层参数偏离初始预训练值
    'freeze_backbone': True,  # 冻结骨干网络，仅微调最后一层，配合蒸馏可达到极高的稳定性
    'sim_anchor_ratio': 1.0,  # 仿真锚点配比
    'missing_class_anchor': 800,  # 缺失实采数据的类别的仿真注入量

    # ---- 训练超参数 ----
    'epochs': 30,  # 稳定微调通常 20-30 轮即可完成对齐
    'batch_size': 128,
    'head_lr': 5e-4,  # 分类输出头的微调学习率
}

MODEL_NAME = FT_CONFIG['model_name']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Mode: Stable Fine-tuning [{MODEL_NAME}]")

if MODEL_NAME == 'transformer':
    from model_transformer_v2 import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn_v2 import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. 数据加载与处理逻辑                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def load_real_data(root_dir, classes_list):
    real_x_list, real_y_list = [], []
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
    return np.stack(real_x_list, axis=0), np.array(real_y_list, dtype=np.int64)


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
# ║  3. 主微调控制逻辑（融合 LwF 知识蒸馏与权重弹簧约束）                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
def main():
    print("Loading Base Simulation Dataset...")
    sim_data = np.load(FT_CONFIG['sim_dataset_path'], allow_pickle=True)
    classes = list(sim_data['classes'])

    test_raw = sim_data['X_test']
    test_labels_idx = sim_data['Y_test']
    Y_test_onehot = np.eye(len(classes), dtype='uint8')[test_labels_idx]
    Z_test = sim_data['snr_test'].reshape(-1, 1)

    X_real, Y_real = load_real_data(FT_CONFIG['real_dataset_root'], classes)

    # 动态构建混合域平衡训练集
    sim_train_raw = sim_data['X_train']
    sim_train_labels = sim_data['Y_train']
    anchor_x_list, anchor_y_list = [], []

    print("\n[数据集构建] 正在生成混合域训练集...")
    for cls_idx, cls_name in enumerate(classes):
        cls_real_mask = (Y_real == cls_idx) if Y_real is not None else np.array([])
        n_real = cls_real_mask.sum()
        cls_sim_indices = np.where(sim_train_labels == cls_idx)[0]
        high_snr_indices = cls_sim_indices[len(cls_sim_indices) // 2:]

        if n_real > 0:
            real_x = X_real[cls_real_mask]
            real_y = Y_real[cls_real_mask]
            n_anchors = int(n_real * FT_CONFIG['sim_anchor_ratio'])
            sampled = np.random.choice(high_snr_indices, size=min(n_anchors, len(high_snr_indices)), replace=False)
            anchor_x_list.extend([sim_train_raw[sampled], real_x])
            anchor_y_list.extend([sim_train_labels[sampled], real_y])
            print(f"  -> {cls_name:8s}: 混合训练 (实采={n_real} 样本, 仿真锚点={len(sampled)} 样本)")
        else:
            n_anchors = FT_CONFIG['missing_class_anchor']
            sampled = np.random.choice(high_snr_indices, size=min(n_anchors, len(high_snr_indices)), replace=False)
            anchor_x_list.append(sim_train_raw[sampled])
            anchor_y_list.append(sim_train_labels[sampled])
            print(f"  -> {cls_name:8s}: 纯仿真锚点填充 (仿真锚点={len(sampled)} 样本)")

    X_ft_train = np.concatenate(anchor_x_list, axis=0)
    Y_ft_train = np.concatenate(anchor_y_list, axis=0)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_ft_train, Y_ft_train, test_size=0.1, random_state=42, stratify=Y_ft_train
    )

    train_loader = DataLoader(IQDataset(X_train, Y_train, normalize=True, augment=True),
                              batch_size=FT_CONFIG['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(IQDataset(X_val, Y_val, normalize=True, augment=False), batch_size=FT_CONFIG['batch_size'],
                            shuffle=False, pin_memory=True)
    test_loader = DataLoader(IQDataset(test_raw, test_labels_idx, normalize=True, augment=False),
                             batch_size=FT_CONFIG['batch_size'], shuffle=False, pin_memory=True)

    # 实例化正在微调的模型（学生模型）
    model = build_model(classes=len(classes)).to(device)
    model.load_state_dict(torch.load(FT_CONFIG['pretrain_weights'], map_location=device))

    # 实例化一个完全冻结的预训练模型作为教师，提供 A, B, C 决策边界锚定
    teacher_model = build_model(classes=len(classes)).to(device)
    teacher_model.load_state_dict(torch.load(FT_CONFIG['pretrain_weights'], map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 骨干网络参数冻结控制
    if FT_CONFIG['freeze_backbone']:
        print("\n[冻结策略] 骨干特征提取网络已硬冻结。")
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name or 'linear' in name or 'out' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_CONFIG['head_lr'],
                                weight_decay=1e-4)
    else:
        print("\n[冻结策略] 骨干特征提取网络未硬冻结。")
        optimizer = optim.AdamW(model.parameters(), lr=FT_CONFIG['head_lr'], weight_decay=1e-4)

    # 软标签与硬标签损失均衡
    criterion_task = nn.CrossEntropyLoss(label_smoothing=0.01)

    nb_epoch = FT_CONFIG['epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epoch, eta_min=1e-7)
    scaler = GradScaler()

    best_val_acc = 0.0
    print("\n[开始稳定微调] 正在结合知识蒸馏和 L2-SP 参数惩罚训练分类头...")

    for epoch in range(nb_epoch):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epoch}")

        for inputs, labels in train_pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 同时前向传播学生网络与冻结的教师网络
            with autocast():
                outputs = model(inputs)
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                # 1. 任务损失（引导模型学习新信号 D 特征）
                loss_task = criterion_task(outputs, labels)

                # 2. 知识蒸馏损失（约束模型保留对 A, B, C 的旧知识判决）
                T = FT_CONFIG['kd_temperature']
                p_teacher = F.softmax(teacher_outputs / T, dim=1)
                p_student = F.log_softmax(outputs / T, dim=1)
                loss_kd = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T ** 2)

                # 3. L2-SP 参数正则化损失（限制分类层权重偏离初始值，充当弹簧约束）
                loss_sp = 0.0
                if FT_CONFIG['l2_sp_lambda'] > 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            ref_param = teacher_model.state_dict()[name]
                            loss_sp += torch.sum((param - ref_param) ** 2)

                # 联合多约束损失函数
                alpha = FT_CONFIG['kd_alpha']
                loss = (1 - alpha) * loss_task + alpha * loss_kd + FT_CONFIG['l2_sp_lambda'] * loss_sp

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

        # 验证
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
            print(f"  ✓ Saved stabilized weights. ({best_val_acc:.4f} → {avg_val_acc:.4f})")
            torch.save(model.state_dict(), FT_CONFIG['save_weights_path'])
            best_val_acc = avg_val_acc

    # 测试
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

    # 总体准确率与混淆矩阵
    cm, right, wrong = mltools.calculate_confusion_matrix(Y_test_onehot, test_Y_hat, classes)
    acc = round(1.0 * right / (right + wrong), 4)
    print(f'\n=== Stabilized Fine-tuned Model Accuracy: {acc * 100:.2f}% ===')
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename=FT_CONFIG['save_confusion_path'])

    # 逐 QAM 类别细化评估
    print(f"\n{'=' * 40}")
    print("QAM Per-Class Accuracy Evaluation (After Distillation):")
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