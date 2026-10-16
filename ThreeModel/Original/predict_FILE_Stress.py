import os
import glob
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import h5py
# 确保 CNN.py 在同一目录下，或者在 Python 路径中
import CNN as mcl  

# ============================================================
# 1. 配置参数
# ============================================================
classes = ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK']

MODEL_WEIGHTS_PATH = 'weights/weights_cnn_mat_a_3.pt'
TEST_DATA_FOLDER = 'E:/Project/AMR-Benchmark/output_mat_files_2FSK/'
NUM_SAMPLES_TO_TEST = 100
FILE_EXTENSION = '.mat'

# --- 压力测试关键参数 ---
STRESS_TEST_ROUNDS = 20       # 每个样本重复测试多少次
PERTURBATION_ROTATION = 60.0   # 最大随机旋转角度 (+/- 5度)

# [核心修改] 使用 SNR (dB) 来控制噪声，而不是固定标准差
# 20 dB = 信号很清晰 (轻微干扰)
# 15 dB = 信号有明显噪点 (标准压力测试) <-- 推荐
# 10 dB = 信号较脏 (高强度测试)
STRESS_SNR_DB = 15.0          

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 2. 核心处理函数
# ============================================================

def per_sample_normalize_single(signal_data):
    """
    归一化到 [-1, 1]。
    注意：必须在添加噪声之后调用！
    """
    min_val = np.min(signal_data)
    max_val = np.max(signal_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(signal_data)
    normalized_data = 2 * (signal_data - min_val) / range_val - 1
    return normalized_data

def add_perturbation_snr(signal_data, max_rotation_deg, target_snr_db):
    """
    [智能扰动函数]
    1. 随机相位旋转
    2. 根据目标信噪比 (SNR) 动态添加噪声，解决信号幅度不一致的问题
    """
    # 复制数据，保护原数据
    perturbed_signal = signal_data.copy()
    
    # 提取 I/Q 并转为复数
    # signal_data shape: (1, 2, 1024)
    I = perturbed_signal[0, 0, :]
    Q = perturbed_signal[0, 1, :]
    s_complex = I + 1j * Q
    
    # --- A. 随机旋转 ---
    rotation_angle = (np.random.rand() * 2 - 1) * max_rotation_deg
    s_rotated = s_complex * np.exp(1j * np.deg2rad(rotation_angle))
    
    # --- B. 基于 SNR 添加噪声 ---
    # 1. 计算信号平均功率 P_signal = Mean(|x|^2)
    sig_power = np.mean(np.abs(s_rotated)**2)
    
    # 2. 计算噪声功率 P_noise
    # SNR_dB = 10 * log10(P_signal / P_noise)  =>  P_noise = P_signal / 10^(SNR/10)
    noise_power = sig_power / (10 ** (target_snr_db / 10))
    
    # 3. 计算噪声标准差 sigma
    # 复噪声功率 P_n = 2 * sigma^2  =>  sigma = sqrt(P_n / 2)
    noise_std = np.sqrt(noise_power / 2)
    
    # 4. 生成噪声
    noise = np.random.normal(0, noise_std, s_rotated.shape) + \
            1j * np.random.normal(0, noise_std, s_rotated.shape)
            
    s_final = s_rotated + noise
    
    # --- C. 还原数据 ---
    perturbed_signal[0, 0, :] = np.real(s_final)
    perturbed_signal[0, 1, :] = np.imag(s_final)
    
    return perturbed_signal

def predict_single_signal(model, signal_data, device, class_list):
    """
    输入信号 -> 归一化 -> 模型推理
    """
    # 1. 维度检查
    if signal_data.shape != (1, 2, 1024):
        if signal_data.shape == (2, 1024):
             signal_data = signal_data.reshape(1, 2, 1024)
        else:
            raise ValueError(f"Shape error: {signal_data.shape}")

    # 2. 归一化 (关键步骤：在预测前一刻进行)
    norm_signal = per_sample_normalize_single(signal_data)

    # 3. 转 Tensor
    model.eval()
    input_tensor = torch.from_numpy(norm_signal).float()
    
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0) # (1, 1, 2, 1024)
        
    input_tensor = input_tensor.to(device)

    # 4. 推理
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)
    confidence, predicted_idx_tensor = torch.max(probabilities, 1)

    return class_list[predicted_idx_tensor.item()], confidence.item()

# ============================================================
# 3. 主程序
# ============================================================
if __name__ == "__main__":
    # A. 加载模型
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Weights not found at {MODEL_WEIGHTS_PATH}")
        exit()
    
    print("Loading model...")
    model = mcl.build_model(classes=len(classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    
    # B. 扫描文件
    print(f"\nScanning '{TEST_DATA_FOLDER}'...")
    search_pattern = os.path.join(TEST_DATA_FOLDER, f'*{FILE_EXTENSION}')
    all_files = glob.glob(search_pattern)
    files_to_test = all_files[:NUM_SAMPLES_TO_TEST]
    
    if not files_to_test:
        print("No files found.")
        exit()

    print("\n" + "=" * 60)
    print(f"STARTING SMART STRESS TEST (SNR Based)")
    print(f"Rounds: {STRESS_TEST_ROUNDS} | Rotation: +/-{PERTURBATION_ROTATION} deg | Target SNR: {STRESS_SNR_DB} dB")
    print("=" * 60)

    total_samples = 0
    stable_samples = 0 
    
    for i, file_path in enumerate(files_to_test):
        filename = os.path.basename(file_path)
        try:
            # 1. 读取原始数据 (Raw Data)
            with h5py.File(file_path, 'r') as f:
                raw_signal = f['data_save'][:] 
            
            # 形状修正
            if raw_signal.shape == (2, 1024):
                raw_signal = raw_signal.reshape(1, 2, 1024)
            elif raw_signal.shape != (1, 2, 1024):
                continue

            # 2. 基准预测 (Baseline) - 没有任何额外噪声
            baseline_pred, baseline_conf = predict_single_signal(model, raw_signal, device, classes)
            
            # 3. 压力测试循环
            consistent_count = 0
            confidences = []
            
            for _ in range(STRESS_TEST_ROUNDS):
                # [关键] 使用基于 SNR 的扰动
                perturbed_signal = add_perturbation_snr(
                    raw_signal, 
                    max_rotation_deg=PERTURBATION_ROTATION, 
                    target_snr_db=STRESS_SNR_DB
                )
                
                # 预测 (函数内部会做归一化)
                p_pred, p_conf = predict_single_signal(model, perturbed_signal, device, classes)
                
                confidences.append(p_conf)
                if p_pred == baseline_pred:
                    consistent_count += 1
            
            # 4. 统计
            stability_rate = (consistent_count / STRESS_TEST_ROUNDS) * 100
            avg_conf = np.mean(confidences) * 100
            
            status = "STABLE" if stability_rate >= 90 else "UNSTABLE"
            
            print(f"\nSample {i+1}: {filename}")
            print(f"  -> Baseline: {baseline_pred} ({baseline_conf*100:.1f}%)")
            print(f"  -> Stress (SNR={STRESS_SNR_DB}dB): {status} ({consistent_count}/{STRESS_TEST_ROUNDS})")
            print(f"  -> Avg Conf: {avg_conf:.1f}%")
            
            if stability_rate >= 90:
                stable_samples += 1
            total_samples += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # C. 总结
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    if total_samples > 0:
        robustness = (stable_samples/total_samples)*100
        print(f"Total Samples: {total_samples}")
        print(f"Target SNR:    {STRESS_SNR_DB} dB")
        print(f"Robust Samples: {stable_samples}")
        print(f"Robustness Rate: {robustness:.1f}%")
        
        if robustness > 90:
            print("\n结论: 模型非常稳健！不仅能抗旋转，也能在 15dB 信噪比下保持准确。")
        elif robustness > 70:
            print("\n结论: 模型比较稳健，但对强噪声开始敏感。")
        else:
            print("\n结论: 模型在噪声下不稳定，可能需要更多低 SNR 数据的训练。")
    else:
        print("No samples processed.")