# -*- coding: utf-8 -*-
"""
test_denoise.py
载入实测数据文件夹中的原始 .mat 文件，经过去噪网络处理后，
输出一模一样的数据集格式（1, 2, 1024），并输出去噪前后的1对1对比星座图。
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from tqdm import tqdm

from model_denoise import IQDenoiseNet

# --- 配置区 ---
SIGNAL_LENGTH = 1024

# 原始实测文件夹路径
INPUT_MAT_DIR = 'E:/Project/AMR-Benchmark/output_mat_files_128QAM_DDC'
# 去噪输出文件夹 (直接输入到你的分类网络中)
OUTPUT_MAT_DIR = 'E:/Project/AMR-Benchmark/output_mat_files_128QAM_DDC_denoised'
# 对比评估图保存路径
PLOT_OUTPUT_DIR = 'E:/Project/AMR-Benchmark/output_plot_reports_128QAM_DDC_denoised'

DENOISE_WEIGHTS_PATH = './weights/denoise_best.pt'
NUM_DIAGNOSTIC_PLOTS = 10  # 要输出对比图的样本数

os.makedirs(OUTPUT_MAT_DIR, exist_ok=True)
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def power_normalize(signal_data):
    """保持和你的分类训练/推理完全对齐的归一化"""
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)

def save_denoise_comparison_plot(before, after, index, filename):
    plt.ioff()
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Real-world IQ Denoise Diagnostics | File: {filename}", fontsize=16)

    b_i, b_q = before[0, :], before[1, :]
    a_i, a_q = after[0, :], after[1, :]

    # 1.1 原始时域
    axs[0, 0].plot(b_i, label='Raw I', alpha=0.7)
    axs[0, 0].plot(b_q, label='Raw Q', alpha=0.7)
    axs[0, 0].set_title('BEFORE: Raw I/Q Time Domain')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 1.2 原始星座
    axs[0, 1].scatter(b_i, b_q, s=6, alpha=0.6, color='darkred')
    axs[0, 1].set_title('BEFORE: Constellation Diagram')
    axs[0, 1].grid(True)
    axs[0, 1].set_aspect('equal', 'box')
    axs[0, 1].set_xlim(-2.2, 2.2)
    axs[0, 1].set_ylim(-2.2, 2.2)

    # 2.1 去噪时域
    axs[1, 0].plot(a_i, label='Cleaned I', alpha=0.7, color='tab:green')
    axs[1, 0].plot(a_q, label='Cleaned Q', alpha=0.7, color='tab:orange')
    axs[1, 0].set_title('AFTER: Denoised I/Q Time Domain')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # 2.2 去噪星座
    axs[1, 1].scatter(a_i, a_q, s=10, alpha=0.8, color='darkblue')
    axs[1, 1].set_title('AFTER: Recovered Constellation')
    axs[1, 1].grid(True)
    axs[1, 1].set_aspect('equal', 'box')
    axs[1, 1].set_xlim(-2.2, 2.2)
    axs[1, 1].set_ylim(-2.2, 2.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(PLOT_OUTPUT_DIR, f'compare_{index:04d}.png')
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    if not os.path.exists(DENOISE_WEIGHTS_PATH):
        print(f"Error: Denoise weights not found at: {DENOISE_WEIGHTS_PATH}")
        exit()

    # 加载去噪网络并初始化
    model = IQDenoiseNet().to(device)
    model.load_state_dict(torch.load(DENOISE_WEIGHTS_PATH, map_location=device))
    model.eval()

    all_files = glob.glob(os.path.join(INPUT_MAT_DIR, '*.mat'))
    if not all_files:
        print("Error: No .mat files found in input directory.")
        exit()

    print(f"Found {len(all_files)} files. Running IQ-Denoise Pipeline...\n")

    for i, file_path in enumerate(tqdm(all_files, desc="Denoising & Packaging")):
        try:
            # 1. 读取原 H5-format 结构中的实测信号数据
            with h5py.File(file_path, 'r') as f:
                raw_signal = f['data_save'][:]  # 形状为 (1, 2, 1024)

            # 2. 进行功率归一化
            normalized_signal = power_normalize(raw_signal.squeeze(0))  # 转换成 (2, 1024)

            # 3. 升维打包喂入网络进行特征重建
            input_tensor = torch.from_numpy(normalized_signal).unsqueeze(0).to(device)

            with torch.no_grad():
                denoised_tensor = model(input_tensor)

            denoised_signal = denoised_tensor.squeeze(0).cpu().numpy()  # 转回 (2, 1024)

            # 再次功率归一化（使去噪后的物理功率完全满足分类器输入条件）
            denoised_signal = power_normalize(denoised_signal)

            # 4. 重点对齐！升维为分类网络相同的 (1, 2, 1024) 形状
            signal_to_save = denoised_signal[np.newaxis, :, :]

            # 5. 保存回标准 H5 导出结构
            filename = os.path.basename(file_path)
            export_path = os.path.join(OUTPUT_MAT_DIR, filename)
            with h5py.File(export_path, 'w') as f_out:
                f_out.create_dataset('data_save', data=signal_to_save)

            # 6. 保存诊断对比图像
            if i < NUM_DIAGNOSTIC_PLOTS:
                save_denoise_comparison_plot(
                    before=normalized_signal,
                    after=denoised_signal,
                    index=i,
                    filename=filename
                )

        except Exception as e:
            print(f"\n  [Error] Failed to process {os.path.basename(file_path)}: {e}")

    print("\n" + "=" * 55)
    print("✓ 实测数据集去噪与标准格式重构打包结束!")
    print(f"  -> 干净数据集输出地址: '{OUTPUT_MAT_DIR}' (直接输入给分类网络预测!)")
    print(f"  -> 1对1对比图报告地址: '{PLOT_OUTPUT_DIR}'")
    print("=" * 55)