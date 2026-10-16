# data_generator_rad.py
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
# [修改] R+A+D 三通道专用目录
DATASET_ROOT = "./dataset_tri_RAD" 
SIM_TRAIN_DIR = os.path.join(DATASET_ROOT, "sim_train")
SIM_VAL_DIR = os.path.join(DATASET_ROOT, "sim_val")
REAL_TEST_DIR = os.path.join(DATASET_ROOT, "real_test")

# 12 类标准名称
CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]

# 真实数据映射
FILE_MAPPING = {
    "E:/Project/AMR-Benchmark/PSK8-03-24-25-10h54m48s14.iq": "8PSK",
    "E:/Project/AMR-Benchmark/QAM16-03-24-25-10h52m09s762.iq": "16QAM",
    "E:/Project/AMR-Benchmark/QAM64-03-24-25-10h51m51s805.iq": "64QAM",
    "E:/Project/AMR-Benchmark/ASK2-03-24-25-10h56m09s501.iq": "2ASK",
    "E:/Project/AMR-Benchmark/BPSK-03-24-25-10h52m32s937.iq": "BPSK",
    "E:/Project/AMR-Benchmark/QPSK-03-24-25-10h53m20s46.iq": "QPSK",
    "E:/Project/AMR-Benchmark/FSK2DEV625KHZ-03-24-25-11h08m50s462.iq": "FSK2",
    "E:/Project/AMR-Benchmark/OQPSK-03-24-25-10h54m02s6.iq": "OQPSK",
    "E:/Project/AMR-Benchmark/PI4DQPSK-03-24-25-10h54m23s982.iq": "PI4DQPSK"
}

SAMPLES_PER_IMAGE = 10240
IMAGE_SIZE = 64
TOTAL_SIM_SAMPLES_PER_CLASS = 800
TRAIN_VAL_RATIO = 0.8
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ================= 1. 数学与图像工具 =================

def fast_hist2d(x, y, bins=64, range_bounds=((-1, 1), (-1, 1))):
    x_min, x_max = range_bounds[0]
    y_min, y_max = range_bounds[1]
    x = torch.clamp(x, x_min, x_max)
    y = torch.clamp(y, y_min, y_max)
    x_idx = ((x - x_min) / (x_max - x_min) * (bins - 1)).long()
    y_idx = ((y - y_min) / (y_max - y_min) * (bins - 1)).long()
    flat_idx = y_idx * bins + x_idx
    hist = torch.bincount(flat_idx, minlength=bins*bins)
    return hist.view(bins, bins).float()

def generate_manifold_heatmap(iq_complex_data, img_size=64):
    """
    计算 R, A, D 三个通道
    Output: (3, H, W)
    """
    if iq_complex_data.dim() == 1:
        iq_complex_data = iq_complex_data.unsqueeze(0)
    iq_data = iq_complex_data.squeeze()
    if iq_data.numel() < 100: return torch.zeros(3, img_size, img_size)

    # --- 1. 基础归一化 ---
    scale = torch.sqrt(torch.mean(iq_data.real**2 + iq_data.imag**2)) + 1e-9
    norm_iq = iq_data / (3.0 * scale)
    
    # --- R 通道 (I vs Q) ---
    real = norm_iq.real
    imag = norm_iq.imag
    
    # --- A 通道 (Phase vs Mag) ---
    phase = torch.angle(norm_iq) / np.pi 
    mag = torch.abs(norm_iq)
    
    # --- D 通道 (Differential I vs Differential Q) ---
    # D[t] = S[t] * conj(S[t-1])
    diff_signal = norm_iq[1:] * torch.conj(norm_iq[:-1])
    # 差分后幅度会变（大约平方倍），需要重新鲁棒归一化
    d_scale = torch.sqrt(torch.mean(diff_signal.real**2 + diff_signal.imag**2)) + 1e-9
    diff_signal = diff_signal / (3.0 * d_scale)
    
    d_real = diff_signal.real
    d_imag = diff_signal.imag
    
    # --- 生成直方图 ---
    bins = img_size
    # R: [-1.2, 1.2]
    hist_r = fast_hist2d(real, imag, bins=bins, range_bounds=((-1.2, 1.2), (-1.2, 1.2)))
    # A: Phase[-1,1], Mag[0, 1.5]
    hist_a = fast_hist2d(phase, mag, bins=bins, range_bounds=((-1.0, 1.0), (0.0, 1.5)))
    # D: [-1.2, 1.2] (差分星座图)
    hist_d = fast_hist2d(d_real, d_imag, bins=bins, range_bounds=((-1.2, 1.2), (-1.2, 1.2)))
    
    # --- 对数压缩 + 归一化 ---
    def process(h):
        h = torch.log1p(h)
        return (h - h.min()) / (h.max() - h.min() + 1e-8)
        
    # 堆叠 R, A, D
    img_tensor = torch.stack([process(hist_r), process(hist_a), process(hist_d)], dim=0) 
    
    return torch.flip(img_tensor, [1]) # Y轴向上

def save_tensor_as_rgb_png(tensor, path):
    """保存为 RGB PNG (Ch1=R, Ch2=G, Ch3=B) -> 对应我们的 (R, A, D)"""
    tensor = tensor.detach().cpu()
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode='RGB').save(path)

# ================= 2. 信号仿真核心 (高保真+全类别) =================

def generate_rrc_filter(N_taps, beta, sps):
    t = torch.arange(-(N_taps - 1) // 2, (N_taps - 1) // 2 + 1, dtype=torch.float32) / sps
    mid_idx = (N_taps - 1) // 2
    t[mid_idx] = 1e-8
    num = torch.sin(np.pi * t * (1 - beta)) + 4 * beta * t * torch.cos(np.pi * t * (1 + beta))
    den = np.pi * t * (1 - (4 * beta * t)**2)
    coeffs = num / den
    coeffs[mid_idx] = 1 - beta + (4 * beta / np.pi)
    return coeffs / torch.sqrt(torch.sum(coeffs**2))

def apply_channel_effects(sig, snr_db):
    N = len(sig)
    freq_offset = np.random.uniform(-50, 50) #-50,50
    t = torch.arange(N, dtype=torch.float32) / 80e3
    phase_offset = np.random.uniform(0, 2*np.pi)
    rotation = torch.exp(1j * (2 * np.pi * freq_offset * t + phase_offset))
    sig = sig * rotation
    
    phase_jitter = torch.randn(N) * 0.005 
    sig = sig * torch.exp(1j * phase_jitter)
    
    g_imbalance = np.random.uniform(0.98, 1.02)
    p_imbalance = np.random.uniform(-1, 1) * np.pi/180
    real = sig.real * g_imbalance
    imag = sig.imag * np.cos(p_imbalance) - sig.real * np.sin(p_imbalance)
    sig = torch.complex(real, imag)
    
    sig_power = torch.mean(torch.abs(sig)**2)
    real_snr = snr_db + 5 
    noise_power = sig_power / (10**(real_snr/10.0))
    noise = (torch.randn_like(sig) + 1j*torch.randn_like(sig)) * torch.sqrt(noise_power/2)
    return sig + noise

# 辅助函数
def get_32qam_constellation():
    points = []
    for x in [-5, -3, -1, 1, 3, 5]:
        for y in [-5, -3, -1, 1, 3, 5]:
            if not (abs(x) == 5 and abs(y) == 5): 
                points.append(x + 1j*y)
    return torch.tensor(points, dtype=torch.complex64)

def get_apsk_symbols(M, ratios, num_symbols):
    if M == 16:
        R1, R2 = 1.0, ratios[0]
        pts_inner = R1 * torch.exp(1j * (torch.arange(4)*2*np.pi/4 + np.pi/4))
        pts_outer = R2 * torch.exp(1j * (torch.arange(12)*2*np.pi/12))
        constellation = torch.cat([pts_inner, pts_outer])
    elif M == 32:
        R1, R2, R3 = 1.0, ratios[0], ratios[1]
        pts_inner = R1 * torch.exp(1j * (torch.arange(4)*2*np.pi/4 + np.pi/4))
        pts_mid = R2 * torch.exp(1j * (torch.arange(12)*2*np.pi/12))
        pts_outer = R3 * torch.exp(1j * (torch.arange(16)*2*np.pi/16))
        constellation = torch.cat([pts_inner, pts_mid, pts_outer])
    indices = torch.randint(0, M, (num_symbols,))
    return constellation[indices]

def generate_sim_signal(mod_type, N, snr_db):
    Fs, Rs = 80e3, 20e3
    sps = int(Fs / Rs)
    num_symbols = int(N / sps) + 50
    
    syms = None
    if mod_type == "BPSK":
        syms = 2*torch.randint(0, 2, (num_symbols,)) - 1 + 0j
    elif mod_type == "QPSK":
        idx = torch.randint(0, 4, (num_symbols,))
        syms = torch.exp(1j * (np.pi/4 + idx * np.pi/2))
    elif mod_type == "8PSK":
        idx = torch.randint(0, 8, (num_symbols,))
        syms = torch.exp(1j * idx * 2*np.pi/8)
    elif mod_type == "OQPSK":
        idx = torch.randint(0, 4, (num_symbols,))
        syms = torch.exp(1j * (np.pi/4 + idx * np.pi/2))
    elif mod_type == "PI4DQPSK":
        bits = torch.randint(0, 4, (num_symbols,))
        phase_shifts = torch.tensor([np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4])[bits]
        phases = torch.cumsum(phase_shifts, dim=0)
        syms = torch.exp(1j * phases)
    elif mod_type == "2ASK":
        bits = torch.randint(0, 2, (num_symbols,)).float()
        syms = bits + 0j
    elif mod_type == "16QAM":
        idx = torch.randint(0, 16, (num_symbols,))
        i = (idx // 4) * 2 - 3
        q = (idx % 4) * 2 - 3
        syms = i + 1j*q
    elif mod_type == "32QAM":
        constellation = get_32qam_constellation()
        idx = torch.randint(0, 32, (num_symbols,))
        syms = constellation[idx]
    elif mod_type == "64QAM":
        idx = torch.randint(0, 64, (num_symbols,))
        i = (idx // 8) * 2 - 7
        q = (idx % 8) * 2 - 7
        syms = i + 1j*q
    elif mod_type == "16APSK":
        syms = get_apsk_symbols(16, [2.7], num_symbols)
    elif mod_type == "32APSK":
        syms = get_apsk_symbols(32, [2.6, 4.5], num_symbols)
    elif mod_type == "FSK2":
        bits = torch.randint(0, 2, (N + 100,))
        f_dev = np.random.uniform(500, 6000) 
        freqs = (bits * 2 - 1) * f_dev
        phase = 2 * np.pi * torch.cumsum(freqs/Fs, dim=0)
        sig = torch.exp(1j * phase)[:N]
        return apply_channel_effects(sig, snr_db)

    upsampled = torch.zeros(num_symbols * sps, dtype=torch.complex64)
    if mod_type == "OQPSK":
        upsampled_i = torch.zeros(num_symbols * sps)
        upsampled_q = torch.zeros(num_symbols * sps)
        upsampled_i[::sps] = syms.real
        q_temp = torch.zeros(num_symbols * sps)
        q_temp[::sps] = syms.imag
        upsampled_q = torch.roll(q_temp, shifts=sps//2)
        upsampled_q[:sps//2] = 0
        upsampled = upsampled_i + 1j * upsampled_q
    else:
        upsampled[::sps] = syms

    taps = 10 * sps + 1
    h = generate_rrc_filter(taps, 0.35, sps)
    real = F.conv1d(upsampled.real.view(1,1,-1), h.view(1,1,-1), padding=taps//2)
    imag = F.conv1d(upsampled.imag.view(1,1,-1), h.view(1,1,-1), padding=taps//2)
    sig = (real + 1j*imag).squeeze()
    
    if len(sig) > N: sig = sig[:N]
    else: sig = F.pad(sig, (0, N - len(sig)))
    return apply_channel_effects(sig, snr_db)

# ================= 3. 执行生成 =================

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        print("Cleaning old dataset...")
        shutil.rmtree(DATASET_ROOT)
    
    print("===== 1. Generating Simulated Data (R+A+D Tri-Channel) =====")
    for cls_name in CLASS_NAMES:
        os.makedirs(os.path.join(SIM_TRAIN_DIR, cls_name), exist_ok=True)
        os.makedirs(os.path.join(SIM_VAL_DIR, cls_name), exist_ok=True)
        print(f"Generating {cls_name}...")
        for i in tqdm(range(TOTAL_SIM_SAMPLES_PER_CLASS)):
            target_dir = SIM_TRAIN_DIR if np.random.rand() < TRAIN_VAL_RATIO else SIM_VAL_DIR
            target_dir = os.path.join(target_dir, cls_name)
            
            snr = np.random.uniform(-5, 25)
            sig = generate_sim_signal(cls_name, SAMPLES_PER_IMAGE, snr)
            heatmap = generate_manifold_heatmap(sig.to(DEVICE), IMAGE_SIZE)
            
            save_path = os.path.join(target_dir, f"{cls_name}_{i:04d}_{snr:.1f}.png")
            save_tensor_as_rgb_png(heatmap, save_path)

    print("\n===== 2. Processing Real Data (R+A+D) =====")
    for file_path, label in FILE_MAPPING.items():
        if not os.path.exists(file_path): 
            print(f"Skip: {file_path}")
            continue
        print(f"Processing {label}...")
        save_dir = os.path.join(REAL_TEST_DIR, label)
        os.makedirs(save_dir, exist_ok=True)
        try:
            raw = np.fromfile(file_path, dtype=np.int16)
            if len(raw) % 2 != 0: raw = raw[:-1]
            I = raw[::2].astype(np.float32)
            Q = raw[1::2].astype(np.float32)
            
            # 给整数加上 [-0.5, 0.5] 的随机小数，填补整数之间的空隙
            I += np.random.uniform(-1.5, 1.5, I.shape)
            Q += np.random.uniform(-1.5, 1.5, Q.shape)

            #iq_complex = (I + 1j * Q).astype(np.complex64) / 32767.0
            iq_complex = (I + 1j * Q).astype(np.complex64) 
            
            num_slices = len(iq_complex) // SAMPLES_PER_IMAGE
            for i in range(num_slices):
                chunk = iq_complex[i*SAMPLES_PER_IMAGE : (i+1)*SAMPLES_PER_IMAGE]
                chunk_tensor = torch.from_numpy(chunk).to(DEVICE)
                heatmap = generate_manifold_heatmap(chunk_tensor, IMAGE_SIZE)
                save_tensor_as_rgb_png(heatmap, os.path.join(save_dir, f"real_{i:04d}_unknown.png"))
        except Exception as e:
            print(f"Error: {e}")
            
    print("\nData generation complete!")