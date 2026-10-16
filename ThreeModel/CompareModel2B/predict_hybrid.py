"""
predict_hybrid.py
=================
混合输入模型推理脚本

用法与原 predict.py 相同：
  · 配置 MODEL_WEIGHTS_PATH 和 TEST_DATA_FOLDER
  · python predict_hybrid.py

改动：
  · 单样本推理时，在内存中实时提取 Branch B/C 特征，无图片文件
  · 支持 use_branch_b / use_branch_c 开关（与训练时保持一致）
"""

import os
import glob
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import h5py

from hybrid_features import extract_all, FEAT_DIM, INST_CHANNELS
from model_hybrid_tcn import build_model

# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════
USE_BRANCH_B = True
USE_BRANCH_C = False

classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

MODEL_WEIGHTS_PATH = 'weights/hybrid_tcn_uc.pt'
TEST_DATA_FOLDER   = 'E:/Project/AMR-Benchmark/output_mat_files_32QAM_DDC/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION      = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Branch B: {USE_BRANCH_B}  Branch C: {USE_BRANCH_C}")


# ══════════════════════════════════════════════════════════════════════════
# 预处理（与训练完全一致）
# ══════════════════════════════════════════════════════════════════════════

def power_normalize(iq2L: np.ndarray) -> np.ndarray:
    """iq2L: (2, L)"""
    pwr = np.mean(iq2L[0] ** 2 + iq2L[1] ** 2)
    if pwr < 1e-10:
        return iq2L.astype(np.float32)
    return (iq2L / np.sqrt(pwr)).astype(np.float32)


def prepare_inputs(iq2L: np.ndarray, device):
    """
    iq2L: (2, 1024)  numpy, 已归一化
    返回: (iq_tensor, inst_tensor, cyclic_tensor) 各 batch=1 的 Tensor
    """
    # Branch A
    iq_tensor = torch.from_numpy(iq2L).float().permute(1, 0).unsqueeze(0).to(device)
    # (1, 1024, 2)

    iq_N = iq2L[np.newaxis, ...]   # (1, 2, 1024)
    inst_np, cyc_np = extract_all(iq_N)

    inst_tensor   = torch.from_numpy(inst_np).float().to(device)    # (1, 3, 1024)
    cyclic_tensor = torch.from_numpy(cyc_np).float().to(device)     # (1, FEAT_DIM)

    return iq_tensor, inst_tensor, cyclic_tensor


# ══════════════════════════════════════════════════════════════════════════
# 单样本推理
# ══════════════════════════════════════════════════════════════════════════

def predict_single(model, iq2L: np.ndarray, device, class_list,
                   temperature: float = 1.0):
    """
    iq2L: (2, 1024)  numpy float32  (已归一化)
    返回: (class_name, confidence)
    """
    model.eval()
    iq_t, inst_t, cyc_t = prepare_inputs(iq2L, device)

    inst_arg  = inst_t  if USE_BRANCH_B else None
    cyclic_arg = cyc_t  if USE_BRANCH_C else None

    with torch.no_grad():
        out  = model(iq_t, inst_arg, cyclic_arg)
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

    model = build_model(
        classes      = len(classes),
        use_branch_b = USE_BRANCH_B,
        use_branch_c = USE_BRANCH_C,
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    all_files = glob.glob(os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}'))
    if not all_files:
        print(f"No '{FILE_EXTENSION}' files found in {TEST_DATA_FOLDER}")
        raise SystemExit(1)

    files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
    print(f"Testing {len(files_to_test)} samples...\n")

    counts = Counter()
    for i, fp in enumerate(files_to_test):
        try:
            with h5py.File(fp, 'r') as f:
                signal = f['data_save'][:]    # (1, 2, 1024)

            if signal.shape != (1, 2, 1024):
                print(f"  [Skip] {os.path.basename(fp)}: shape {signal.shape}")
                continue

            iq2L = power_normalize(signal.squeeze(0))   # (2, 1024)
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
