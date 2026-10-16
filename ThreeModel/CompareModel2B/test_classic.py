"""
test_classic.py
---------------
使用经典骨干网络对实测 .mat 文件进行批量调制识别推理。

修改方式：只需改动 § CONFIG 中的变量
  MODEL_NAME        : 与训练时保持一致
  USE_PRETRAINED    : 与训练时保持一致（决定权重文件名）
  MODEL_WEIGHTS_PATH: 也可直接指定完整路径覆盖自动生成的路径
"""

import os
import glob
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import h5py

from classic_models import build_classic_model


# ============================================================
# § CONFIG —— 只需修改这里
# ============================================================
MODEL_NAME         = 'resnet50'          # resnet50 | efficientnet_b3 | mobilenet_v3_large | densenet121
USE_PRETRAINED     = True                # 与训练时一致
MODEL_WEIGHTS_PATH = ''                  # 留空则自动按命名规则查找；或填完整路径手动指定

TEST_DATA_FOLDER   = 'E:/Project/AMR-Benchmark/output_mat_files_QAM256/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION     = '.mat'
# ============================================================

# 类别列表（与训练时完全一致）
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']

# 自动推断权重路径
if not MODEL_WEIGHTS_PATH:
    pretrain_tag       = 'pretrained' if USE_PRETRAINED else 'scratch'
    MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_{pretrain_tag}_SionnaA_6.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")
print(f"Model        : {MODEL_NAME}  (pretrained={USE_PRETRAINED})")
print(f"Weights path : {MODEL_WEIGHTS_PATH}")


# ============================================================
# 1. 功率归一化（必须与 train_classic.py 完全一致）
# ============================================================
def power_normalize(signal_data: np.ndarray) -> np.ndarray:
    """
    输入 shape: (1, 2, 1024) 或 (2, 1024)
    """
    if signal_data.ndim == 3:
        I = signal_data[0, 0, :]
        Q = signal_data[0, 1, :]
    else:
        I, Q = signal_data[0], signal_data[1]

    power = np.mean(I**2 + Q**2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


# ============================================================
# 2. 单信号推理函数
# ============================================================
def predict_single_signal(model, signal_data: np.ndarray, device, class_list):
    """
    对单个信号进行调制识别预测。

    Args:
        signal_data : shape (1, 2, 1024)，float32，已功率归一化
    Returns:
        (predicted_class: str, confidence: float)

    维度变换：
        (1, 2, 1024)
          → squeeze(0) → (2, 1024)
          → permute(1,0) → (1024, 2)   ← 与 Dataset.__getitem__ 一致
          → unsqueeze(0) → (1, 1024, 2) ← 批次维度
    """
    if signal_data.shape != (1, 2, 1024):
        raise ValueError(f"Expected shape (1, 2, 1024), got {signal_data.shape}")

    model.eval()

    sample       = signal_data.squeeze(0)               # (2, 1024)
    input_tensor = torch.from_numpy(sample).float()     # (2, 1024)
    input_tensor = input_tensor.permute(1, 0)           # (1024, 2)
    input_tensor = input_tensor.unsqueeze(0).to(device) # (1, 1024, 2)

    with torch.no_grad():
        output = model(input_tensor)                    # (1, num_classes)

    probs      = F.softmax(output, dim=1)
    conf, idx  = torch.max(probs, 1)

    return class_list[idx.item()], conf.item()


# ============================================================
# 3. 主执行
# ============================================================
if __name__ == '__main__':
    # 检查权重文件
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"[Error] Model weights not found: '{MODEL_WEIGHTS_PATH}'")
        print("  → 请先运行 train_classic.py 完成训练，或手动设置 MODEL_WEIGHTS_PATH")
        exit(1)

    # 加载模型
    print("\nLoading model architecture...")
    model = build_classic_model(
        model_name  = MODEL_NAME,
        num_classes = len(classes),
        pretrained  = False   # 推理时不需要下载 ImageNet 权重，直接加载训练好的权重
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.\n")

    # 查找测试文件
    print(f"Scanning '{TEST_DATA_FOLDER}' for '{FILE_EXTENSION}' files...")
    all_files    = glob.glob(os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}'))
    if not all_files:
        print(f"[Error] No '{FILE_EXTENSION}' files found.")
        exit(1)

    files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
    print(f"Found {len(all_files)} files. Testing first {len(files_to_test)}.\n")

    # 批量推理
    prediction_counts = Counter()
    print("=" * 55)
    print("  BATCH PREDICTION")
    print("=" * 55)

    for i, file_path in enumerate(files_to_test):
        fname = os.path.basename(file_path)
        print(f"\n[{i+1:>3}/{len(files_to_test)}] {fname}")

        try:
            # 读取 .mat 文件（HDF5 格式）
            with h5py.File(file_path, 'r') as f:
                raw = f['data_save'][:]              # (1, 2, 1024)

            if raw.shape != (1, 2, 1024):
                print(f"  [Skip] Unexpected shape: {raw.shape}")
                continue

            # 功率归一化（与训练完全一致）
            normalized = power_normalize(raw)

            # 推理
            pred, conf = predict_single_signal(model, normalized, device, classes)

            print(f"  → Predicted : {pred}")
            print(f"  → Confidence: {conf*100:.2f}%")
            prediction_counts[pred] += 1

        except FileNotFoundError:
            print(f"  [Error] File not found: {fname}")
        except Exception as e:
            print(f"  [Error] {fname}: {e}")

    # 汇总
    print("\n" + "=" * 55)
    print("  PREDICTION SUMMARY")
    print("=" * 55)
    total_processed = sum(prediction_counts.values())
    print(f"Processed: {total_processed} / {len(files_to_test)}\n")

    if prediction_counts:
        print(f"{'Class':<14}  {'Count':>6}  {'Ratio':>7}")
        print("-" * 32)
        for cls in classes:
            cnt = prediction_counts.get(cls, 0)
            pct = cnt / total_processed * 100 if total_processed else 0
            bar = '█' * int(pct / 5)   # 简单文本条形图
            print(f"  {cls:<12}  {cnt:>5}   {pct:5.1f}%  {bar}")

        best = prediction_counts.most_common(1)[0]
        print(f"\n  Most predicted: {best[0]}  ({best[1]} times, "
              f"{best[1]/total_processed*100:.1f}%)")
    else:
        print("No successful predictions.")

    print("=" * 55)
    print("Done.")
