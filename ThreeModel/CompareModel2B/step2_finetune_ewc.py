"""
step2_finetune_ewc.py
基于预训练权重的 EWC 微调，防止灾难性遗忘

运行前确认：
  1. step1_gen_finetune_data.py 已成功生成 ./finetune_v2_dataset/finetune_v2.npz
  2. PRETRAINED_PATH 指向你的预训练权重
  3. ORIGINAL_NPZ 指向原始大数据集（用于计算 Fisher 矩阵和重放）
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ★ 修改这里的路径和超参数                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
PRETRAINED_PATH = 'weights/tcn_v2_SionnaA_6.pt'
ORIGINAL_NPZ    = './sionna_dataset/SionnaA_6.npz'     # 原始大数据集
FINETUNE_NPZ    = './finetune_v2_dataset/finetune_v2.npz'
OUTPUT_PATH     = 'weights/tcn_v2_finetune_ewc.pt'

# --- 超参数 ---
REPLAY_RATIO    = 0.6     # 从原始数据集抽取的重放比例（相对于微调集大小）
                          # 0.6 = 新数据40% + 原始重放60%
                          # 如果遗忘严重，调高到0.8；如果128QAM没改善，调低到0.4
EWC_LAMBDA      = 1000000     # EWC惩罚强度
                          # 训练日志中 EWC损失/CE损失 应在1~5倍之间
                          # 比值>10 → 降到100；比值<0.1 → 升到1000
N_FISHER        = 4000    # 用于估算Fisher矩阵的原始数据样本数（越多越准，越慢）
EPOCHS          = 20
LR              = 2e-4    # 比从头训练低一个数量级
BATCH_SIZE      = 256
NUM_WORKERS     = 4       # DataLoader workers，Windows上若报错改为0

CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM',
           '128QAM', '256QAM', 'MSK']

# ════════════════════════════════════════════════════════════════════════
#  模型导入（根据你的实际文件名修改）
# ════════════════════════════════════════════════════════════════════════
from model_tcn_v2 import build_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")


# ════════════════════════════════════════════════════════════════════════
#  EWC 核心
# ════════════════════════════════════════════════════════════════════════

def compute_fisher(model, dataloader, device, n_samples):
    """
    用原始数据子集估计 Fisher 信息矩阵的对角近似。
    Fisher_i ≈ E[(∂logP/∂θ_i)²]
    必须在修改模型参数之前调用。
    """
    print(f"  计算 Fisher 矩阵（使用 {n_samples} 个样本）...")
    model.eval()
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}

    count = 0
    for X, Y in dataloader:
        if count >= n_samples:
            break
        X, Y = X.to(device), Y.to(device)
        model.zero_grad()
        out  = model(X)
        loss = nn.CrossEntropyLoss()(out, Y)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        count += len(Y)

    for n in fisher:
        fisher[n] /= max(count, 1)

    print(f"  Fisher 矩阵计算完成（实际使用 {count} 个样本）")
    return fisher


def ewc_penalty(model, fisher, optimal_params, ewc_lambda):
    """
    EWC 惩罚项：λ/2 * Σ F_i * (θ_i - θ*_i)²
    """
    loss = torch.tensor(0.0, device=device)
    for n, p in model.named_parameters():
        if n in fisher and n in optimal_params:
            loss = loss + (fisher[n] * (p - optimal_params[n]) ** 2).sum()
    return ewc_lambda / 2.0 * loss


# ════════════════════════════════════════════════════════════════════════
#  数据加载
# ════════════════════════════════════════════════════════════════════════

def to_model_input(X_np):
    """(N, 2, L) → (N, L, 2)，模型期望的输入格式"""
    return torch.from_numpy(X_np).permute(0, 2, 1).float()


def build_mixed_dataset(original_npz, finetune_npz, replay_ratio):
    """
    混合数据集 = 微调数据集 + 从原始集随机抽取的重放样本
    replay_ratio：重放样本数 / 微调样本数
    """
    print("\n构建混合数据集...")
    fine = np.load(finetune_npz, allow_pickle=True)
    orig = np.load(original_npz, allow_pickle=True)

    n_fine  = len(fine['Y_train'])
    n_replay = int(n_fine * replay_ratio)
    n_replay = min(n_replay, len(orig['Y_train']))

    idx = np.random.choice(len(orig['Y_train']), n_replay, replace=False)
    X_replay = orig['X_train'][idx]
    Y_replay = orig['Y_train'][idx]

    X_mixed = np.concatenate([fine['X_train'], X_replay], axis=0)
    Y_mixed = np.concatenate([fine['Y_train'], Y_replay], axis=0)

    perm = np.random.permutation(len(Y_mixed))
    X_mixed, Y_mixed = X_mixed[perm], Y_mixed[perm]

    print(f"  微调数据: {n_fine:,}  重放数据: {n_replay:,}  总计: {len(Y_mixed):,}")
    print(f"  验证集  : {len(fine['Y_test']):,}（仅使用微调集验证集）")

    return (X_mixed, Y_mixed,
            fine['X_test'], fine['Y_test'])


# ════════════════════════════════════════════════════════════════════════
#  训练 / 验证
# ════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, fisher, optimal_params):
    model.train()
    total_ce = total_ewc = correct = total = 0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        out = model(X)

        loss_ce  = criterion(out, Y)
        loss_ewc = ewc_penalty(model, fisher, optimal_params, EWC_LAMBDA)
        loss     = loss_ce + loss_ewc

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = len(Y)
        total_ce  += loss_ce.item()  * bs
        total_ewc += loss_ewc.item() * bs
        correct   += (out.argmax(1) == Y).sum().item()
        total     += bs

    return total_ce / total, total_ewc / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    correct = total = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        correct += (model(X).argmax(1) == Y).sum().item()
        total   += len(Y)
    return correct / total


@torch.no_grad()
def eval_per_class(model, loader, class_names):
    """每类单独统计准确率，用于监控是否发生遗忘"""
    model.eval()
    n = len(class_names)
    correct_per = [0] * n
    total_per   = [0] * n
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        preds = model(X).argmax(1)
        for c in range(n):
            mask = (Y == c)
            correct_per[c] += (preds[mask] == c).sum().item()
            total_per[c]   += mask.sum().item()
    return [correct_per[c] / max(total_per[c], 1) for c in range(n)]


# ════════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # ── 检查文件 ──
    for path, name in [(PRETRAINED_PATH, '预训练权重'),
                       (ORIGINAL_NPZ,    '原始数据集'),
                       (FINETUNE_NPZ,    '微调数据集')]:
        if not os.path.exists(path):
            print(f"错误：{name} 不存在 → {path}")
            return

    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else '.', exist_ok=True)

    # ── 1. 加载预训练模型 ──
    print(f"\n加载预训练模型: {PRETRAINED_PATH}")
    model = build_model(classes=len(CLASSES))
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # ── 2. 计算 Fisher 矩阵（必须在修改模型前！）──
    orig = np.load(ORIGINAL_NPZ, allow_pickle=True)
    n_f  = min(N_FISHER, len(orig['X_train']))
    idx_f = np.random.choice(len(orig['X_train']), n_f, replace=False)
    Xf = to_model_input(orig['X_train'][idx_f])
    Yf = torch.from_numpy(orig['Y_train'][idx_f])
    fisher_loader = DataLoader(TensorDataset(Xf, Yf),
                               batch_size=128, shuffle=False)
    fisher_dict = compute_fisher(model, fisher_loader, device, N_FISHER)

    # 保存最优参数（即预训练参数，EWC的锚点）
    optimal_params = {n: p.detach().clone().to(device)
                      for n, p in model.named_parameters() if p.requires_grad}
    del orig, Xf, Yf   # 释放内存

    # ── 3. 构建混合数据集 ──
    X_tr, Y_tr, X_te, Y_te = build_mixed_dataset(
        ORIGINAL_NPZ, FINETUNE_NPZ, REPLAY_RATIO)

    train_loader = DataLoader(
        TensorDataset(to_model_input(X_tr), torch.from_numpy(Y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(
        TensorDataset(to_model_input(X_te), torch.from_numpy(Y_te)),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))

    # ── 4. 优化器和调度器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 20)
    criterion = nn.CrossEntropyLoss()

    # ── 5. 训练循环 ──
    best_acc = 0.0
    print(f"\n{'Ep':>4} {'TrainAcc':>9} {'ValAcc':>9} "
          f"{'CE_Loss':>9} {'EWC_Loss':>10} {'Ratio':>7} {'LR':>9}")
    print("─" * 65)

    for ep in range(1, EPOCHS + 1):
        ce, ewc_val, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, fisher_dict, optimal_params)
        val_acc = eval_epoch(model, test_loader)
        scheduler.step()

        ratio  = ewc_val / (ce + 1e-8)
        lr_now = scheduler.get_last_lr()[0]

        # EWC/CE 比值警告
        flag = ''
        if ratio > 10:
            flag = ' ← EWC太强,降低EWC_LAMBDA'
        elif ratio < 0.05:
            flag = ' ← EWC太弱,提高EWC_LAMBDA'

        print(f"{ep:>4d} {tr_acc:>9.4f} {val_acc:>9.4f} "
              f"{ce:>9.4f} {ewc_val:>10.2f} {ratio:>7.2f} {lr_now:>9.2e}{flag}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_PATH)

    # ── 6. 每类准确率（最终模型）──
    print(f"\n最优验证准确率: {best_acc:.4f}  →  {OUTPUT_PATH}")
    print("\n加载最优模型，统计每类准确率...")
    model.load_state_dict(torch.load(OUTPUT_PATH, map_location=device))
    per_class_acc = eval_per_class(model, test_loader, CLASSES)

    print(f"\n{'类别':<14} {'准确率':>8}")
    print("─" * 25)
    for cls, acc in zip(CLASSES, per_class_acc):
        flag = ' ← 关注' if acc < 0.6 else ''
        print(f"  {cls:<12} {acc*100:>7.1f}%{flag}")
    print(f"\n  平均: {np.mean(per_class_acc)*100:.1f}%")
    print(f"\n下一步：运行 step3_evaluate.py 在实测数据上验证")


if __name__ == '__main__':
    main()
