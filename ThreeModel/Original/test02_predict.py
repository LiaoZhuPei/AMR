import os
import glob
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from test01_model import build_model

classes = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

MODEL_WEIGHTS_PATH = 'weights/cnn_bilstm_attn_minmax.pt'
TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_GMSK3/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION = '.mat'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===== 归一化必须与训练一致：[-1, 1] min-max =====
def per_sample_normalize_single(signal_data):
    """
    输入 shape: (2, 1024)
    与训练时完全一致的归一化
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data, dtype=np.float32)
    normalized_data = 2 * (signal_data - min_val) / range_val - 1
    return normalized_data.astype(np.float32)


def predict_single_signal(model, signal_data, device, class_list, temperature=1.0):
    """
    输入: signal_data shape (1, 2, 1024)
    模型期望: (batch, 1024, 2)
    """
    if signal_data.shape != (1, 2, 1024):
        raise ValueError(f"Expected shape (1, 2, 1024), got {signal_data.shape}")

    model.eval()
    sample = signal_data.squeeze(0)                       # (2, 1024)
    input_tensor = torch.from_numpy(sample).float()
    input_tensor = input_tensor.permute(1, 0)             # (1024, 2)
    input_tensor = input_tensor.unsqueeze(0).to(device)   # (1, 1024, 2)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output / temperature, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

    return class_list[predicted_idx.item()], confidence.item()


if __name__ == "__main__":
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Weights not found at '{MODEL_WEIGHTS_PATH}'")
        exit()

    model = build_model(classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    print("Model loaded.")

    search_pattern = os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}')
    all_files = glob.glob(search_pattern)
    if not all_files:
        print(f"No '{FILE_EXTENSION}' files found.")
        exit()

    files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
    print(f"Testing {len(files_to_test)} samples...\n")

    prediction_counts = Counter()

    for i, file_path in enumerate(files_to_test):
        try:
            with h5py.File(file_path, 'r') as f:
                signal = f['data_save'][:]

            if signal.shape != (1, 2, 1024):
                continue

            # ===== 使用 [-1,1] 归一化 =====
            normalized = per_sample_normalize_single(signal.squeeze(0))
            normalized = normalized.reshape(1, 2, 1024)

            pred_class, conf = predict_single_signal(model, normalized, device, classes)
            print(f"  [{i+1:3d}] {os.path.basename(file_path)}: "
                  f"{pred_class:<12} ({conf*100:.1f}%)")
            prediction_counts[pred_class] += 1

        except Exception as e:
            print(f"  [Error] {os.path.basename(file_path)}: {e}")

    # 统计
    print(f"\n{'='*50}")
    total = sum(prediction_counts.values())
    for cls in classes:
        cnt = prediction_counts.get(cls, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {cls:<12}: {cnt:>4} ({pct:5.1f}%)")

    if prediction_counts:
        top = prediction_counts.most_common(1)[0]
        print(f"\n  Most predicted: {top[0]} ({top[1]} times, {top[1]/total*100:.1f}%)")
    print(f"{'='*50}")