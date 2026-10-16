import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import Counter
import LSTM as lstm

# ==============================================================================
# --- 1. 配置区域 (最终确认版) ---
# ==============================================================================
classes = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '16APSK', '32APSK', '64QAM', '32QAM', 'AM', 'FM']

HISARMOD_CLASSES_MAP = {
    0: 'BPSK', 1: 'QPSK', 2: '8PSK', 3: '16PSK', 4: '32PSK', 5: '64PSK',
    6: '4QAM', 7: '8QAM', 8: '16QAM', 9: '32QAM', 10: '64QAM', 11: '128QAM',
    12: '256QAM', 13: '2FSK', 14: '4FSK', 15: '8FSK', 16: '16FSK',
    17: '4PAM', 18: '8PAM', 19: '16PAM', 20: 'AM-DSB', 21: 'AM-DSB-SC',
    22: 'AM-USB', 23: 'AM-LSB', 24: 'FM', 25: 'PM'
}

HISARMOD_AM_VARIANTS = ['AM-DSB', 'AM-DSB-SC', 'AM-USB', 'AM-LSB']
HISARMOD_FM_VARIANTS = ['FM']
CLASSES_TO_TEST_HISARMOD_NAMES = HISARMOD_AM_VARIANTS + HISARMOD_FM_VARIANTS

MODEL_WEIGHTS_PATH = 'weights/finetuned_lstm_mode_A.pt'
HISARMOD_TEST_DATA_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/HisarModdataset/Test/test.mat'
HISARMOD_TEST_LABELS_PATH = 'E:/Project/AMR-Benchmark/HisarMod/dataset/HisarModdataset/Test/test_labels1.csv'

# --- 修改点 1: 更新 DATA_KEY ---
DATA_KEY = 'data_save'

NUM_SAMPLES_PER_CLASS = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ... 核心函数 (per_sample_normalize_single, predict_single_signal) 不变 ...
def per_sample_normalize_single(signal_data):
    min_val, max_val = np.min(signal_data), np.max(signal_data)
    range_val = max_val - min_val
    return np.zeros_like(signal_data) if range_val == 0 else 2 * (signal_data - min_val) / range_val - 1


def predict_single_signal(model, signal_data, device, class_list):
    if signal_data.shape != (1, 2, 1024):
        raise ValueError(f"Expected signal shape (1, 2, 1024), got {signal_data.shape}")
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(signal_data).float().permute(0, 2, 1).to(device)
        output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_idx_tensor = torch.max(probabilities, 1)
    return class_list[predicted_idx_tensor.item()], confidence.item()


# ==============================================================================
# --- 3. 主执行程序 ---
# ==============================================================================
if __name__ == "__main__":
    print(f"Using device: {device}")

    # --- a. 加载模型 ---
    try:
        print("Loading LSTM model architecture...")
        model = lstm.build_model(classes=len(classes))
        print(f"Loading trained weights from '{MODEL_WEIGHTS_PATH}'...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # --- b. 加载并筛选 HisarMod 数据 ---
    print("\nLoading and filtering HisarMod dataset...")
    try:
        with h5py.File(HISARMOD_TEST_DATA_PATH, 'r') as f:
            raw_signals = f[DATA_KEY][:]
            print(f"Successfully loaded data with original shape: {raw_signals.shape}")

        # --- 修改点 2: 添加正确的转置操作 ---
        all_test_signals = np.transpose(raw_signals, (2, 1, 0))
        print(f"Transposed data to new shape: {all_test_signals.shape}")

        labels_df = pd.read_csv(HISARMOD_TEST_LABELS_PATH, header=None)
        all_test_labels_idx = labels_df.iloc[:, 0].values

        all_test_labels_str = np.array([HISARMOD_CLASSES_MAP.get(idx, 'Unknown') for idx in all_test_labels_idx])

        print(f"Loaded {len(all_test_signals)} signals and {len(all_test_labels_str)} labels.")
        if len(all_test_signals) != len(all_test_labels_str):
            raise ValueError("Mismatch in number of samples!")

        signals_to_test, true_labels_to_test = [], []

        for class_name_hisarmod in CLASSES_TO_TEST_HISARMOD_NAMES:
            indices = np.where(all_test_labels_str == class_name_hisarmod)[0]
            if not len(indices):
                print(f"Warning: No samples found for class '{class_name_hisarmod}'.")
                continue

            if NUM_SAMPLES_PER_CLASS is not None:
                indices = indices[:NUM_SAMPLES_PER_CLASS]

            print(f"Selected {len(indices)} samples for class '{class_name_hisarmod}'.")
            signals_to_test.append(all_test_signals[indices])

            eval_label = 'AM' if class_name_hisarmod in HISARMOD_AM_VARIANTS else 'FM'
            true_labels_to_test.extend([eval_label] * len(indices))

        if not signals_to_test:
            print("Error: No valid signals found for the specified classes. Exiting.")
            exit()
        signals_to_test = np.vstack(signals_to_test)

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error during data loading: {e}")
        print("Please double-check file paths and the DATA_KEY in the script.")
        exit()

    # --- c & d. 预测与评估 ---
    correct_predictions, total_samples = Counter(), Counter()

    print("\n" + "=" * 50 + "\nSTARTING BATCH PREDICTION\n" + "=" * 50)

    for i in range(len(signals_to_test)):
        true_class_name = true_labels_to_test[i]
        sample = np.expand_dims(signals_to_test[i], axis=0)
        normalized_sample = per_sample_normalize_single(sample)

        predicted_class, confidence = predict_single_signal(model, normalized_sample, device, classes)

        is_correct = "✔" if predicted_class == true_class_name else "✖"
        print(
            f"Sample {i + 1:>4}/{len(signals_to_test)} | True: {true_class_name:<5} | Predicted: {predicted_class:<5} | Conf: {confidence * 100:6.2f}% | {is_correct}")

        total_samples[true_class_name] += 1
        if predicted_class == true_class_name:
            correct_predictions[true_class_name] += 1

    print("\n" + "=" * 50 + "\n                PREDICTION SUMMARY\n" + "=" * 50)
    overall_correct, overall_total = 0, 0

    for class_name in ['AM', 'FM']:
        correct = correct_predictions.get(class_name, 0)
        total = total_samples.get(class_name, 0)
        overall_correct += correct
        overall_total += total

        if total > 0:
            accuracy = (correct / total) * 100
            print(f"- Class: {class_name:<10} | Accuracy: {accuracy:.2f}% ({correct}/{total})")
        else:
            print(f"- Class: {class_name:<10} | No samples were tested.")

    if overall_total > 0:
        overall_accuracy = (overall_correct / overall_total) * 100
        print("-" * 50)
        print(f"  Overall Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")
    print("=" * 50)