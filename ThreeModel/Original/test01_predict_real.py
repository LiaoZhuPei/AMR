import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ===== 导入您的模型 =====
from test01_model import build_model

# --- 基本配置 ---
classes = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

MODEL_WEIGHTS_PATH = 'weights/cnn_bilstm_attn_matb1_1.pt'

# [关键变更] 直接读取刚才生成的数据集，而不是 mat 文件夹
TEST_DATA_NPZ = 'E:/Project/AMR-Benchmark/HisarMod/Pytorch_Sequence/ModelCompare/sionna_dataset/Sionna_real_1.npz'
BATCH_SIZE = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def power_normalize_batch(signal_data):
    """
    功率归一化：将信号平均功率归一化为 1
    输入 shape: (2, 1024) 或 (1, 2, 1024)
    必须与训练时的预处理完全一致！
    """
    I = signal_data[..., 0, :] if signal_data.ndim == 3 else signal_data[0]
    Q = signal_data[..., 1, :] if signal_data.ndim == 3 else signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)



if __name__ == "__main__":
    # 1. 检查文件
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")
    if not os.path.exists(TEST_DATA_NPZ):
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATA_NPZ}. Run generate_data_pro.py first.")

    # 2. 构建并加载模型
    print("Loading model architecture...")
    model = build_model(classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. 加载 NPZ 数据
    print(f"Loading test data from {TEST_DATA_NPZ}...")
    data = np.load(TEST_DATA_NPZ)
    X_test = data['X_test']  # Shape: (N, 2, 1024)
    Y_test = data['Y_test']  # Shape: (N,)
    SNR_test = data['snr_test']  # Shape: (N,)

    print(f"Loaded {len(Y_test)} test samples.")

    # 4. 数据预处理与 DataLoader
    # 归一化
    X_test_norm = power_normalize_batch(X_test)

    # 转换维度: 原(N, 2, 1024) -> 模型需要 (N, 1024, 2)
    X_test_norm = np.transpose(X_test_norm, (0, 2, 1))

    dataset = TensorDataset(torch.from_numpy(X_test_norm).float(),
                            torch.from_numpy(Y_test).long(),
                            torch.from_numpy(SNR_test).float())

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 开始批量推理
    all_preds = []
    all_labels = []
    all_snrs = []

    print("\nStarting inference...")
    with torch.no_grad():
        for batch_x, batch_y, batch_snr in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
            all_snrs.extend(batch_snr.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)

    # 6. 统计结果分析
    overall_acc = np.mean(all_preds == all_labels) * 100
    print("\n" + "=" * 50)
    print(f"OVERALL ACCURACY: {overall_acc:.2f}%")
    print("=" * 50)

    # 按类别统计
    print(f"\n{'Class':<12} | {'Accuracy (%)':<15} | {'Samples'}")
    print("-" * 45)
    for i, cls_name in enumerate(classes):
        idx = (all_labels == i)
        if np.sum(idx) > 0:
            acc = np.mean(all_preds[idx] == all_labels[idx]) * 100
            print(f"{cls_name:<12} | {acc:<15.2f} | {np.sum(idx)}")

    # 按 SNR 统计 (可选：如果您想看高低SNR下的表现差异)
    print("\n" + "=" * 50)
    print(f"{'SNR (dB)':<10} | {'Accuracy (%)':<15}")
    print("-" * 30)
    unique_snrs = sorted(list(set(all_snrs)))
    for snr in unique_snrs:
        idx = (all_snrs == snr)
        acc = np.mean(all_preds[idx] == all_labels[idx]) * 100
        print(f"{int(snr):<10} | {acc:<15.2f}")