"""
predict_cnn.py
==============
特征提取 + SimpleCNN 推理脚本，接口与原 predict.py 一致。
"""

import os
import glob
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import h5py

from feature_extract import extract_features
from model_cnn import build_model

# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

MODEL_WEIGHTS_PATH  = 'weights/simple_cnn_Sionna0_a_2.pt'
TEST_DATA_FOLDER    = 'E:/Project/AMR-Benchmark/output_mat_files_32QAM_DDC/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION      = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ══════════════════════════════════════════════════════════════════════════
# 预处理
# ══════════════════════════════════════════════════════════════════════════

def power_normalize(iq2L: np.ndarray) -> np.ndarray:
    pwr = np.mean(iq2L[0] ** 2 + iq2L[1] ** 2)
    if pwr < 1e-10:
        return iq2L.astype(np.float32)
    return (iq2L / np.sqrt(pwr)).astype(np.float32)


def predict_single(model, iq2L: np.ndarray, device,
                   class_list, temperature: float = 1.0):
    """
    iq2L: (2, 1024) numpy float32，已功率归一化
    """
    model.eval()
    feat = extract_features(iq2L, sps=4)          # (1, 8, 1024)
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
