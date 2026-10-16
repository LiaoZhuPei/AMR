# -*- coding: utf-8 -*-
"""
train_denoise.py
载入具有去噪理想对的 .npz 文件，并开始训练去噪网络。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model_denoise import IQDenoiseNet

# --- 配置区 ---
DATA_PATH = './multirate_dataset/amr_multirate_1.npz'
PLOT_DIR = './train_plots'
WEIGHTS_DIR = './weights'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 载入数据并读取类别列表
data = np.load(DATA_PATH, allow_pickle=True)
classes = list(data['classes'])
print(f"Loaded Classes: {classes}")

X_train_noisy = data['X_train']  # (N, 2, 1024)
X_train_clean = data['X_clean_train']  # (N, 2, 1024)
Y_train = data['Y_train']

X_test_noisy = data['X_test']  # (N_t, 2, 1024)
X_test_clean = data['X_clean_test']  # (N_t, 2, 1024)
Y_test = data['Y_test']


class DenoiseDataset(Dataset):
    def __init__(self, noisy_data, clean_data, labels):
        self.noisy = noisy_data
        self.clean = clean_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # inputs 形状 (2, 1024), targets 形状 (2, 1024)
        return torch.from_numpy(self.noisy[idx]).float(), torch.from_numpy(self.clean[idx]).float(), self.labels[idx]


# 诊断绘图函数
def plot_train_epoch(epoch, noisy, denoised, clean, cls_name):
    plt.ioff()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Epoch {epoch} Denoise Diagnostics | Modulation: {cls_name}", fontsize=14)

    n_c = noisy[0] + 1j * noisy[1]
    d_c = denoised[0] + 1j * denoised[1]
    c_c = clean[0] + 1j * clean[1]

    axs[0].scatter(n_c.real, n_c.imag, s=5, alpha=0.5, color='darkred')
    axs[0].set_title("Input (Distorted & Noisy)")
    axs[0].grid(True)
    axs[0].set_aspect('equal', 'box')

    axs[1].scatter(d_c.real, d_c.imag, s=8, alpha=0.7, color='darkblue')
    axs[1].set_title("Output (Denoised Grid)")
    axs[1].grid(True)
    axs[1].set_aspect('equal', 'box')

    axs[2].scatter(c_c.real, c_c.imag, s=8, alpha=0.8, color='green')
    axs[2].set_title("Target (Ideal Constellation)")
    axs[2].grid(True)
    axs[2].set_aspect('equal', 'box')

    for ax in axs:
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"epoch_{epoch:03d}_{cls_name}.png"))
    plt.close(fig)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training De-noise network on Device: {device}")

    # DataLoader
    train_dataset = DenoiseDataset(X_train_noisy, X_train_clean, Y_train)
    val_dataset = DenoiseDataset(X_test_noisy, X_test_clean, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = IQDenoiseNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs = 40

    print("\n" + "=" * 50)
    print("  IQ-DenoiseNet Supervisor Training initialized")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            train_total += inputs.size(0)

        avg_train_loss = train_loss / train_total

        # 验证和出图
        model.eval()
        val_loss = 0.0
        val_total = 0
        plot_128qam, plot_32qam = False, False

        with torch.no_grad():
            for inputs, targets, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)

                # 每个 Epoch 各自捕获 32QAM 和 128QAM 单样本绘制诊断图
                cls_idx = labels[0].item()
                cls_name = classes[cls_idx]

                if cls_name == '128QAM' and not plot_128qam:
                    plot_train_epoch(epoch, inputs[0].cpu().numpy(), outputs[0].cpu().numpy(), targets[0].cpu().numpy(),
                                     '128QAM')
                    plot_128qam = True
                if cls_name == '32QAM' and not plot_32qam:
                    plot_train_epoch(epoch, inputs[0].cpu().numpy(), outputs[0].cpu().numpy(), targets[0].cpu().numpy(),
                                     '32QAM')
                    plot_32qam = True

        avg_val_loss = val_loss / val_total
        print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 最优保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            weights_path = os.path.join(WEIGHTS_DIR, 'denoise_best.pt')
            torch.save(model.state_dict(), weights_path)
            print(f"  ✓ Saved new best denoise weights with Loss: {best_val_loss:.6f}")

    print("\n✓ 去噪网络训练结束!")