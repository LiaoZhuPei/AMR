"""
test_classic.py  (v2)
---------------------
对实测 .mat 文件进行批量调制识别推理。
§ CONFIG 只需改 MODEL_NAME 和路径。
"""

import os
import glob
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import h5py

from classic_models_1d import build_classic_model

# ============================================================
# § CONFIG
# ============================================================
MODEL_NAME          = 'densenet1d'   # resnet1d | inception1d | densenet1d | tcn
MODEL_WEIGHTS_PATH  = ''           # 留空则自动按命名规则查找
TEST_DATA_FOLDER    = 'E:/Project/AMR-Benchmark/output_mat_files_16QAM_DDC/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION      = '.mat'
# ============================================================

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

if not MODEL_WEIGHTS_PATH:
    MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_Sionna0_b.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
print(f"Model   : {MODEL_NAME}")
print(f"Weights : {MODEL_WEIGHTS_PATH}")


def power_normalize(signal_data: np.ndarray) -> np.ndarray:
    if signal_data.ndim == 3:
        I, Q = signal_data[0, 0, :], signal_data[0, 1, :]
    else:
        I, Q = signal_data[0], signal_data[1]
    power = np.mean(I**2 + Q**2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


def predict_single(model, signal_data: np.ndarray):
    """signal_data shape: (1, 2, 1024)，已归一化"""
    sample = signal_data.squeeze(0)                          # (2,1024)
    x = torch.from_numpy(sample).float().permute(1, 0)      # (1024,2)
    x = x.unsqueeze(0).to(device)                           # (1,1024,2)
    with torch.no_grad():
        out  = model(x)
    prob = F.softmax(out, dim=1)
    conf, idx = torch.max(prob, 1)
    return classes[idx.item()], conf.item()


if __name__ == '__main__':
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"[Error] Weights not found: {MODEL_WEIGHTS_PATH}")
        exit(1)

    model = build_classic_model(MODEL_NAME, num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device).eval()
    print("Model loaded.\n")

    all_files     = glob.glob(os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}'))
    files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
    print(f"Found {len(all_files)} files, testing {len(files_to_test)}.\n")

    counts = Counter()
    print("=" * 55)
    for i, fp in enumerate(files_to_test):
        fname = os.path.basename(fp)
        print(f"[{i+1:>3}/{len(files_to_test)}] {fname}")
        try:
            with h5py.File(fp, 'r') as f:
                raw = f['data_save'][:]
            if raw.shape != (1, 2, 1024):
                print(f"  [Skip] shape={raw.shape}")
                continue
            norm = power_normalize(raw)
            pred, conf = predict_single(model, norm)
            print(f"  → {pred}  ({conf*100:.1f}%)")
            counts[pred] += 1
        except Exception as e:
            print(f"  [Error] {e}")

    total = sum(counts.values())
    print("\n" + "=" * 55)
    print(f"  SUMMARY  ({total}/{len(files_to_test)} processed)")
    print("=" * 55)
    print(f"  {'Class':<14} {'Count':>5}  {'%':>6}  Bar")
    print("  " + "-" * 45)
    for cls in classes:
        n   = counts.get(cls, 0)
        pct = n/total*100 if total else 0
        bar = '█' * int(pct/4)
        print(f"  {cls:<14} {n:>5}   {pct:5.1f}%  {bar}")
    if counts:
        best = counts.most_common(1)[0]
        print(f"\n  Most predicted: {best[0]}  ({best[1]}×, {best[1]/total*100:.1f}%)")
    print("=" * 55)
