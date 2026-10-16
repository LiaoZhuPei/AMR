"""
predict_cnn.py
==============
特征提取 + RegularizedCNN 推理脚本。
增加了对 npz 数据集类别顺序的自动读取逻辑，消除了标签硬编码错位的风险。
"""

import os
import glob
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import h5py

from feature_extract_G import extract_features
from model_cnn_G import build_model

# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════
# 1. 训练集数据路径（用于动态读取 classes 标签顺序，防止硬编码错位）
DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/Pytorch_Sequence/ModelCompare2B/sionna_dataset/Sionna0_a.npz'

# 2. 备用硬编码类别名（如果未找到 DATA_PATH 文件时，此为备用列表）
# 注意：请务必保证此硬编码列表与您训练该权重时采用的 npz 自带 data['classes'] 顺序完全相同
default_classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                   '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

MODEL_WEIGHTS_PATH  = 'weights/simple_cnn_Sionna0_a_g.pt'
TEST_DATA_FOLDER    = 'E:/Project/AMR-Benchmark/output_mat_files_QAM16/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION      = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════════
# 预处理与加载类名
# ══════════════════════════════════════════════════════════════════════════

def load_classes():
    """动态加载数据集中的类别序列，防止标签硬编码导致分类指示错位"""
    if os.path.exists(DATA_PATH):
        try:
            data = np.load(DATA_PATH, allow_pickle=True)
            loaded_classes = list(data['classes'])
            print(f"✓ 成功从数据集动态载入类别顺序 ({len(loaded_classes)} 类):\n  {loaded_classes}")
            return loaded_classes
        except Exception as e:
            print(f"⚠ 加载数据集类别失败: {e}，将使用备用硬编码类别。")
    else:
        print(f"⚠ 未在 '{DATA_PATH}' 找到数据集文件，已退回到默认硬编码类别。")
        print("  请确认您的硬编码类别顺序与训练时模型的类别索引严格一致。")
    return default_classes


def power_normalize(iq2L: np.ndarray) -> np.ndarray:
    """
    对输入信号进行整帧功率归一化（功率标准：1.0）
    """
    pwr = np.mean(iq2L[0] ** 2 + iq2L[1] ** 2)
    if pwr < 1e-10:
        return iq2L.astype(np.float32)
    return (iq2L / np.sqrt(pwr)).astype(np.float32)


def predict_single(model, iq2L: np.ndarray, device,
                   class_list, temperature: float = 1.0):
    """
    单样本推理。
    iq2L: (2, 1024) numpy float32，已功率归一化到总功率 1.0
    """
    model.eval()
    feat = extract_features(iq2L, sps=4)                 # (1, 8, 1024)
    feat_t = torch.from_numpy(feat).float().to(device)   # (1, 8, 1024)
    with torch.no_grad():
        out   = model(feat_t)
    probs = F.softmax(out / temperature, dim=1)
    conf, idx = torch.max(probs, 1)
    return class_list[idx.item()], conf.item()


# ══════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: weights not found at '{MODEL_WEIGHTS_PATH}'")
        raise SystemExit(1)

    classes = load_classes()

    model = build_model(classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    all_files = glob.glob(os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}'))
    if not all_files:
        print(f"No '{FILE_EXTENSION}' files found.")
        raise SystemExit(1)

    files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
    print(f"Testing {len(files_to_test)} samples...\n")

    counts = Counter()
    for i, fp in enumerate(files_to_test):
        try:
            with h5py.File(fp, 'r') as f:
                signal = f['data_save'][:]

            if signal.shape != (1, 2, 1024):
                print(f"  [Skip] {os.path.basename(fp)}: shape {signal.shape}")
                continue

            # 使用与训练集一致的功率归一化（1.0 标准）
            iq2L = power_normalize(signal.squeeze(0))
            pred_class, conf = predict_single(model, iq2L, device, classes)
            print(f"  [{i+1:3d}] {os.path.basename(fp)}: "
                  f"{pred_class:<12} ({conf*100:.1f}%)")
            counts[pred_class] += 1

        except Exception as e:
            print(f"  [Error] {os.path.basename(fp)}: {e}")

    print(f"\n{'='*50}")
    print("  PREDICTION SUMMARY")
    print(f"{'='*50}")
    total = sum(counts.values())
    for cls in classes:
        cnt = counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {cls:<12}: {cnt:>4} ({pct:5.1f}%)")
    if counts:
        top = counts.most_common(1)[0]
        print(f"\n  Most predicted: {top[0]} ({top[1]} times, {top[1]/total*100:.1f}%)")
    print(f"{'='*50}")