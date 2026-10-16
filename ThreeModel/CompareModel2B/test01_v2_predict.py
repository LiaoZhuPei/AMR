"""
统一预测脚本 v2：适配双分支模型
加载 .mat 文件进行单样本预测
"""
import os
import glob
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import h5py

# ╔══════════════════════════════════════════════════════════════╗
# ║  修改这里切换模型和配置                                        ║
# ╚══════════════════════════════════════════════════════════════╝
MODEL_NAME = 'tcn'          # 选项: 'transformer', 'tcn', 'bilstm'
SIGNAL_LENGTH = 1024        # 与训练时的 SIGNAL_LENGTH 一致

# ============================================================
# 导入 v2 双分支模型
# ============================================================
if MODEL_NAME == 'transformer':
    from model_transformer_v2 import build_model
elif MODEL_NAME == 'tcn':
    from model_tcn_v2 import build_model
elif MODEL_NAME == 'bilstm':
    from test01_v2_model import build_model
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}. Choose from 'transformer', 'tcn', 'bilstm'")

# --- 配置 ---
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK']
# PSK8  QAM16 QAM64 QAM256  ASK2 PI4DQPSK
MODEL_WEIGHTS_PATH = f'weights/{MODEL_NAME}_v2_Activate_AMR_Dataset.pt'
TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_128QAM_DDC'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Using model: {MODEL_NAME} (v2 dual-branch)")
print(f"Signal length: {SIGNAL_LENGTH}")


# --- 预处理（必须与训练一致） ---
def power_normalize(signal_data):
    """功率归一化，与训练完全一致"""
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


def predict_single_signal(model, signal_data, device, class_list, temperature=1.0):
    """
    输入: signal_data shape (1, 2, SIGNAL_LENGTH)
    模型期望: (batch, SIGNAL_LENGTH, 2)
    """
    expected_shape = (1, 2, SIGNAL_LENGTH)
    if signal_data.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {signal_data.shape}")

    model.eval()
    sample = signal_data.squeeze(0)                       # (2, L)
    input_tensor = torch.from_numpy(sample).float()
    input_tensor = input_tensor.permute(1, 0)             # (L, 2)
    input_tensor = input_tensor.unsqueeze(0).to(device)   # (1, L, 2)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output / temperature, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

    return class_list[predicted_idx.item()], confidence.item()


# --- 主程序 ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Weights not found at '{MODEL_WEIGHTS_PATH}'")
        exit()

    # 构建 v2 模型并加载权重
    model = build_model(classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{MODEL_NAME}] Parameters: {total_params:,}")

    search_pattern = os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}')
    all_files = glob.glob(search_pattern)
    if not all_files:
        print(f"No '{FILE_EXTENSION}' files found.")
        exit()

    files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
    print(f"Testing {len(files_to_test)} samples with [{MODEL_NAME}_v2]...\n")

    prediction_counts = Counter()

    for i, file_path in enumerate(files_to_test):
        try:
            with h5py.File(file_path, 'r') as f:
                signal = f['data_save'][:]

            expected_shape = (1, 2, SIGNAL_LENGTH)
            if signal.shape != expected_shape:
                print(f"  [Skip] {os.path.basename(file_path)}: "
                      f"shape {signal.shape}, expected {expected_shape}")
                continue

            signal = power_normalize(signal.squeeze(0)).reshape(1, 2, SIGNAL_LENGTH)

            pred_class, conf = predict_single_signal(model, signal, device, classes)

            print(f"  [{i + 1:3d}] {os.path.basename(file_path)}: "
                  f"{pred_class:<12} ({conf * 100:.1f}%)")
            prediction_counts[pred_class] += 1

        except Exception as e:
            print(f"  [Error] {os.path.basename(file_path)}: {e}")

    # 统计
    print(f"\n{'=' * 50}")
    print(f"  [{MODEL_NAME}_v2] PREDICTION SUMMARY")
    print(f"{'=' * 50}")
    total = sum(prediction_counts.values())
    for cls in classes:
        cnt = prediction_counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {cls:<12}: {cnt:>4} ({pct:5.1f}%)")

    if prediction_counts:
        top = prediction_counts.most_common(1)[0]
        print(f"\n  Most predicted: {top[0]} ({top[1]} times, {top[1] / total * 100:.1f}%)")
    print(f"{'=' * 50}")