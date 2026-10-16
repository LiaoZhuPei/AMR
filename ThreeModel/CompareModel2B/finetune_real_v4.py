"""
AMR 极限突破与安全稳定微调脚本 (ultimate_stable_finetune.py)
采用分类选择性蒸馏 (CS-KD) 与 分类选择性权重锚定 (CS-L2-SP)，实现 A,B,C 绝对稳定与 D 性能极限突破
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
# ║  1. CONFIGURATION — 极限微调配置                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
FT_CONFIG = {
    'model_name': 'tcn',
    'pretrain_weights': 'weights/tcn_v2_Sionna0_c.pt',
    'sim_dataset_path': './sionna_dataset/Sionna0_c.npz',
    'real_dataset_root': 'E:/Project/AMR-Benchmark/real_dataset/',
    'save_weights_path': 'weights/tcn_v2_ultimate_finetuned_0c.pt',
    'save_confusion_path': 'figure/tcn_v2_ultimate_finetune_confusion_0c.png',

    # ---- 核心控制：指定需要极限突破性能的问题类别 D ----
    'target_classes': ['128QAM', '256QAM'],  # 在此列表中的类：彻底免除蒸馏限制，放开参数权重，倾斜梯度资源

    # ---- 选择性可塑性超参数 ----
    'kd_alpha': 0.7,  # A, B, C 类别的知识保护权重（值越大，A,B,C越稳定）
    'kd_temperature': 2.0,  # 蒸馏温度
    'l2_sp_lambda': 1e-4,  # 基础权重漂移惩罚
    'freeze_backbone': True,  # 强烈推荐锁定骨干，仅微调最后一层，配合选择性松绑即可使D性能最大化

    'sim_anchor_ratio': 1.0,
    'missing_class_anchor': 800,

    # ---- 训练超参数 ----
    'epochs': 45,
    'batch_size': 128,
    'head_lr': 1e-3,  # 放开D后，可以使用稍大的学习率加速 D 的决策面重构
}

MODEL_NAME = FT_CONFIG['model_name']
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Mode: Ultimate Selective Fine-tuning [{MODEL_NAME}]")

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
# ║  3. 核心控制：分类选择性约束算法                                        ║
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

    # 提取目标突破类别的索引
    target_cls_indices = [classes.index(c) for c in FT_CONFIG['target_classes'] if c in classes]
    print(f"\n[极限性能突破目标定义]: {FT_CONFIG['target_classes']} (索引: {target_cls_indices})")

    # 动态构建混合域训练集
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

    # 实例化学生模型与教师模型
    model = build_model(classes=len(classes)).to(device)
    model.load_state_dict(torch.load(FT_CONFIG['pretrain_weights'], map_location=device))

    teacher_model = build_model(classes=len(classes)).to(device)
    teacher_model.load_state_dict(torch.load(FT_CONFIG['pretrain_weights'], map_location=device))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 骨干特征提取网络硬冻结
    if FT_CONFIG['freeze_backbone']:
        print("\n[冻结策略] 骨干特征网络已硬冻结，仅微调最后一层分类输出头。")
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name or 'linear' in name or 'out' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_CONFIG['head_lr'],
                                weight_decay=1e-4)
    else:
        print("\n[冻结策略] 全网微调开放。")
        optimizer = optim.AdamW(model.parameters(), lr=FT_CONFIG['head_lr'], weight_decay=1e-4)

    # ── 类别权重：大幅度倾斜资源给问题类别 D (128QAM) ──
    class_weights = torch.ones(len(classes), dtype=torch.float32, device=device)
    for idx in target_cls_indices:
        class_weights[idx] = 3.0  # 3.0倍的高重罚权重，强制梯度下降优先拟合 D

    criterion_task = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.01)

    nb_epoch = FT_CONFIG['epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epoch, eta_min=1e-7)
    scaler = GradScaler()

    best_val_acc = 0.0
    print("\n[开始选择性极限微调] A, B, C 被强行锁死锚定，D 彻底解放并进行极限性能拟合...")

    # 转换为 tensor 方便 GPU 内高速判定类别
    target_cls_tensor = torch.tensor(target_cls_indices, device=device)

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
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                # 1. 任务损失（对 128QAM 提供 3.0 倍的高梯度流刺激）
                loss_task = criterion_task(outputs, labels)

                # 2. 【核心创新】：分类选择性知识蒸馏 (CS-KD)
                # 找出非目标类（即需要保护的 A, B, C 样本）
                kd_mask = ~torch.isin(labels, target_cls_tensor)

                if kd_mask.any():
                    # 仅对 A, B, C 样本计算蒸馏惩罚，强行锁定其决策概率
                    T = FT_CONFIG['kd_temperature']
                    p_teacher_sub = F.softmax(teacher_outputs[kd_mask] / T, dim=1)
                    p_student_sub = F.log_softmax(outputs[kd_mask] / T, dim=1)
                    loss_kd = F.kl_div(p_student_sub, p_teacher_sub, reduction='batchmean') * (T ** 2)
                else:
                    loss_kd = 0.0

                # 3. 【核心创新】：分类选择性参数漂移约束 (CS-L2-SP)
                # 分类层偏离初始权重的弹簧约束仅重度作用于 A, B, C
                loss_sp = 0.0
                if FT_CONFIG['l2_sp_lambda'] > 0:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            ref_param = teacher_model.state_dict()[name]

                            # 如果是分类层权重矩阵
                            if 'fc' in name or 'classifier' in name or 'linear' in name or 'out' in name:
                                if param.ndim == 2:  # Weight
                                    for c_idx in range(len(classes)):
                                        if c_idx in target_cls_indices:
                                            # 放开 D 对应的行，使其几乎不受参数约束，可以大范围画出完美边界
                                            loss_sp += 0.05 * torch.sum((param[c_idx] - ref_param[c_idx]) ** 2)
                                        else:
                                            # 重度锚定 A, B, C 对应的行
                                            loss_sp += 1.0 * torch.sum((param[c_idx] - ref_param[c_idx]) ** 2)
                                elif param.ndim == 1:  # Bias
                                    for c_idx in range(len(classes)):
                                        if c_idx in target_cls_indices:
                                            loss_sp += 0.05 * ((param[c_idx] - ref_param[c_idx]) ** 2)
                                        else:
                                            loss_sp += 1.0 * ((param[c_idx] - ref_param[c_idx]) ** 2)
                            else:
                                loss_sp += torch.sum((param - ref_param) ** 2)

                # 多约束联合损失计算（非目标类施加强蒸馏，目标类享受全自适应自由度）
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
            print(f"  ✓ Saved stabilized and adaptive weights. ({best_val_acc:.4f} → {avg_val_acc:.4f})")
            torch.save(model.state_dict(), FT_CONFIG['save_weights_path'])
            best_val_acc = avg_val_acc

    # 测试评估
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
    print(f'\n=== Ultimate Fine-tuned Model Accuracy: {acc * 100:.2f}% ===')
    mltools.plot_confusion_matrix(cm, labels=classes, save_filename=FT_CONFIG['save_confusion_path'])

    # 逐 QAM 类型单独细致统计
    print(f"\n{'=' * 40}")
    print("QAM Per-Class Accuracy Evaluation (After Ultimate Adaptation):")
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