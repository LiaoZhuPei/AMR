import os
import shutil
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# ================= 配置区域 =================
# [修改] 数据集路径 (标记为 NewR_VSG)
DATASET_ROOT = "./dataset_tri_RAD_Tensor_VSG_NewR"
SIM_TRAIN_DIR = os.path.join(DATASET_ROOT, "sim_train")
SIM_VAL_DIR = os.path.join(DATASET_ROOT, "sim_val")
REAL_TEST_DIR = os.path.join(DATASET_ROOT, "real_test")

CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]

# 真实数据路径
FILE_MAPPING = {
    "E:/Project/AMR-Benchmark/PSK8-03-24-25-10h54m48s14.iq": "8PSK",
    "E:/Project/AMR-Benchmark/QAM16-03-24-25-10h52m09s762.iq": "16QAM",
    "E:/Project/AMR-Benchmark/QAM64-03-24-25-10h51m51s805.iq": "64QAM",
    "E:/Project/AMR-Benchmark/ASK2-03-24-25-10h56m09s501.iq": "2ASK",
    "E:/Project/AMR-Benchmark/BPSK-03-24-25-10h52m32s937.iq": "BPSK",
    "E:/Project/AMR-Benchmark/L20250425_103838_F6000000000_S400000_B100000_M0_IQ_ID0_nozaoshen_100ksps_QPSK.dat": "QPSK",
    "E:/Project/AMR-Benchmark/FSK2DEV625KHZ-03-24-25-11h08m50s462.iq": "FSK2",
    "E:/Project/AMR-Benchmark/L20250425_104306_F6000000000_S400000_B100000_M0_IQ_ID0_nojiazao_100ksps_OQPSK.dat": "OQPSK",
    "E:/Project/AMR-Benchmark/PI4DQPSK-03-24-25-10h54m23s982.iq": "PI4DQPSK"
}

SAMPLES_PER_IMAGE = 1024
IMAGE_SIZE = 64
TOTAL_SIM_SAMPLES_PER_CLASS = 800
TRAIN_VAL_RATIO = 0.8
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ================= 1. VSG 仿真器 (保持不变) =================
class LabDeviceSimulator:
    def __init__(self, sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9):
        self.fs = sample_rate
        self.rs = symbol_rate
        self.fc = carrier_freq
        self.sps = int(sample_rate / symbol_rate)
        self.rrc_taps = self._generate_rrc(alpha=0.35, span=10, sps=self.sps)

    def _generate_rrc(self, alpha, span, sps):
        t = np.arange(-span*sps, span*sps+1) / sps
        t[t == 0] = 1e-8
        t[np.abs(t) == 1/(4*alpha)] += 1e-8
        num = np.sin(np.pi*t*(1-alpha)) + 4*alpha*t*np.cos(np.pi*t*(1+alpha))
        den = np.pi*t*(1-(4*alpha*t)**2)
        return num / den

    def generate_packet(self, mod_type, num_samples, ppm_error, iq_gain_db, iq_phase_deg, snr_db):
        num_symbols = num_samples // self.sps + 50
        syms = self._modulate(mod_type, num_symbols)
        
        # 上采样
        sig_upsampled = np.zeros(num_symbols * self.sps, dtype=complex)
        if mod_type == 'OQPSK':
            i_sym = syms.real; q_sym = syms.imag
            sig_i = np.zeros_like(sig_upsampled, dtype=float); sig_i[::self.sps] = i_sym
            sig_q = np.zeros_like(sig_upsampled, dtype=float); sig_q[::self.sps] = q_sym
            sig_q = np.roll(sig_q, self.sps // 2)
            sig_base = sig_i + 1j * sig_q
        else:
            sig_upsampled[::self.sps] = syms
            sig_base = sig_upsampled

        # 硬件损伤
        sig_impaired = self._apply_iq_imbalance(sig_base, iq_gain_db, iq_phase_deg)
        sig_shaped = scipy.signal.convolve(sig_impaired, self.rrc_taps, mode='same')

        # 频偏
        cfo_hz = self.fc * ppm_error * 1e-6
        t = np.arange(len(sig_shaped)) / self.fs
        rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
        sig_rotated = sig_shaped * rotation

        # 噪声
        sig_power = np.mean(np.abs(sig_rotated)**2)
        noise_power = sig_power / (10**(snr_db/10.0))
        noise = (np.random.randn(len(sig_rotated)) + 1j*np.random.randn(len(sig_rotated))) * np.sqrt(noise_power/2)
        
        return (sig_rotated + noise)[:num_samples]

    def _modulate(self, mod_type, n):
        if mod_type == 'QPSK' or mod_type == 'OQPSK':
            return np.exp(1j * (np.pi/4 + np.random.randint(0, 4, n) * np.pi/2))
        elif mod_type == '8PSK':
            return np.exp(1j * np.random.randint(0, 8, n) * 2*np.pi/8)
        elif mod_type == 'BPSK':
            return 2*np.random.randint(0, 2, n) - 1 + 0j
        elif mod_type == '16QAM':
            i = 2*np.random.randint(0, 4, n) - 3; q = 2*np.random.randint(0, 4, n) - 3
            return i + 1j*q
        elif mod_type == '32QAM': 
            i = np.random.randint(-5, 6, n); q = np.random.randint(-5, 6, n)
            return i + 1j*q
        elif mod_type == '64QAM':
            i = 2*np.random.randint(0, 8, n) - 7; q = 2*np.random.randint(0, 8, n) - 7
            return i + 1j*q
        elif mod_type == '16APSK':
            r = np.random.choice([1.0, 2.5], n)
            theta = np.random.uniform(0, 2*np.pi, n)
            return r * np.exp(1j*theta)
        elif mod_type == '32APSK':
            r = np.random.choice([1.0, 2.5, 4.0], n)
            theta = np.random.uniform(0, 2*np.pi, n)
            return r * np.exp(1j*theta)
        elif mod_type == 'FSK2':
            f_dev = 5000 / 100000 
            bits = np.random.randint(0, 2, n)
            freqs = (bits * 2 - 1) * f_dev
            phase = 2 * np.pi * np.cumsum(freqs)
            return np.exp(1j * phase)
        elif mod_type == '2ASK':
            return np.random.randint(0, 2, n).astype(float) + 0j
        elif mod_type == 'PI4DQPSK':
            bits = np.random.randint(0, 4, n)
            shifts = np.array([np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4])[bits]
            phases = np.cumsum(shifts)
            return np.exp(1j * phases)
            
        return np.random.randn(n) + 1j*np.random.randn(n)

    def _apply_iq_imbalance(self, sig, gain_db, phase_deg):
        gain = 10**(gain_db / 20.0)
        phase_rad = phase_deg * np.pi / 180.0
        i_new = sig.real
        q_new = gain * (sig.real * np.sin(phase_rad) + sig.imag * np.cos(phase_rad))
        return i_new + 1j * q_new

# ================= 2. 关键修改：Tensor 生成逻辑 (R=ATM) =================

def fast_hist2d(x, y, bins=64, range_bounds=((-1, 1), (-1, 1))):
    x_min, x_max = range_bounds[0]; y_min, y_max = range_bounds[1]
    x = torch.clamp(x, x_min, x_max); y = torch.clamp(y, y_min, y_max)
    x_idx = ((x - x_min) / (x_max - x_min) * (bins - 1)).long()
    y_idx = ((y - y_min) / (y_max - y_min) * (bins - 1)).long()
    flat_idx = y_idx * bins + x_idx
    hist = torch.bincount(flat_idx, minlength=bins*bins)
    return hist.view(bins, bins).float()

def generate_tensor_data(iq_complex_data, img_size=64):
    """
    生成数据张量 (3, 64, 64)
    Ch1 (R): Amplitude Transition Matrix (ATM) -> 抗频偏
    Ch2 (A): Phase-Amplitude Hist -> 辅助
    Ch3 (D): Diff-Phase Hist (Downsampled) -> 区分 PSK
    """
    if iq_complex_data.dim() == 1:
        iq_complex_data = iq_complex_data.unsqueeze(0)
    iq_data = iq_complex_data.squeeze()
    
    if iq_data.numel() < 100: 
        return torch.zeros(3, img_size, img_size)

    # 1. 归一化
    scale = torch.sqrt(torch.mean(iq_data.real**2 + iq_data.imag**2)) + 1e-9
    norm_iq = iq_data / (3.0 * scale)
    
    # 提取基础特征
    mag = torch.abs(norm_iq)
    phase = torch.angle(norm_iq) / np.pi 
    
    # --- R 通道 [修改]: 幅度转移矩阵 (ATM) ---
    # 原先是 Real vs Imag (星座图)，现在改为 Current Mag vs Prev Mag
    # 这具有绝对的旋转不变性，对 QAM 的层级跳变和 QPSK 的过零点非常敏感
    mag_curr = mag[1:]
    mag_prev = mag[:-1]
    
    # 范围设定：归一化后幅度主要集中在 0 ~ 1.5
    hist_r = fast_hist2d(mag_prev, mag_curr, bins=img_size, range_bounds=((-0.1, 1.5), (-0.1, 1.5)))
    
    # --- A 通道: 相位-幅度 ---
    # 保持原样，提供辅助信息
    hist_a = fast_hist2d(phase, mag, bins=img_size, range_bounds=((-1.0, 1.0), (0.0, 1.5)))
    
    # --- D 通道: 降采样差分 ---
    # 保持之前验证有效的逻辑 (SPS=4 下的差分)
    
    # 1. 寻找最佳采样点
    best_offset = 0
    max_energy = -1.0
    for offset in range(4):
        temp_down = norm_iq[offset::4]
        en = torch.sum(torch.abs(temp_down)**2)
        if en > max_energy:
            max_energy = en
            best_offset = offset
            
    iq_down = norm_iq[best_offset::4]
    
    # 2. 计算差分
    diff_signal = iq_down[1:] * torch.conj(iq_down[:-1])
    
    # 3. 重新归一化
    d_scale = torch.sqrt(torch.mean(diff_signal.real**2 + diff_signal.imag**2)) + 1e-9
    diff_signal = diff_signal / (3.0 * d_scale)
    
    d_real = diff_signal.real
    d_imag = diff_signal.imag
    
    # D 通道直方图
    hist_d = fast_hist2d(d_real, d_imag, bins=img_size, range_bounds=((-1.5, 1.5), (-1.5, 1.5)))
    
    # --- 堆叠与后处理 ---
    def process(h):
        h = torch.log1p(h)
        mx = h.max()
        if mx > 1e-6: h = h / mx
        return h

    tensor = torch.stack([process(hist_r), process(hist_a), process(hist_d)], dim=0)
    return torch.flip(tensor, [1])

# ================= 执行生成 =================
if __name__ == "__main__":

    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    
    print("===== Generating VSG-Sim Tensor (New R-Channel: ATM) =====")
    print("Parameters: SNR 10-25dB, PPM -10~10, IQ Imbal")
    
    sim = LabDeviceSimulator(sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9)
    
    # 1. 仿真数据生成
    for cls_name in CLASS_NAMES:
        os.makedirs(os.path.join(SIM_TRAIN_DIR, cls_name), exist_ok=True)
        os.makedirs(os.path.join(SIM_VAL_DIR, cls_name), exist_ok=True)
        
        print(f"Generating Sim {cls_name}...")
        for i in tqdm(range(TOTAL_SIM_SAMPLES_PER_CLASS)):
            target_dir = SIM_TRAIN_DIR if np.random.rand() < TRAIN_VAL_RATIO else SIM_VAL_DIR
            target_dir = os.path.join(target_dir, cls_name)
            
            # [VSG 参数设置]
            snr = np.random.uniform(10, 25)
            ppm = np.random.uniform(-10, 10)
            gain = np.random.uniform(-0.5, 0.5)
            phase = np.random.uniform(-2, 2)
            
            # 生成信号
            sig = sim.generate_packet(cls_name, SAMPLES_PER_IMAGE, ppm, gain, phase, snr)
            
            if isinstance(sig, np.ndarray): 
                sig_tensor = torch.from_numpy(sig).to(DEVICE)
            
            tensor_data = generate_tensor_data(sig_tensor, IMAGE_SIZE)
            
            save_path = os.path.join(target_dir, f"{cls_name}_{i:04d}_SNR{snr:.1f}.pt")
            torch.save(tensor_data.cpu(), save_path) 
            
            if i == 0:
                from PIL import Image
                arr = (tensor_data.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(target_dir, "viz_sample.png"))

    # 2. 实测数据处理
    print("\nProcessing Real Data...")
    for file_path, label in FILE_MAPPING.items():
        if not os.path.exists(file_path): 
            print(f"Skip {label}: File not found")
            continue
            
        save_dir = os.path.join(REAL_TEST_DIR, label)
        os.makedirs(save_dir, exist_ok=True)
        try:
            if file_path.endswith('.dat'):
                raw = np.fromfile(file_path, dtype='>i2')
                if len(raw) > 1024: raw = raw[1024:]
            else:
                raw = np.fromfile(file_path, dtype=np.int16)
            
            if len(raw) % 2 != 0: raw = raw[:-1]
            
            I = raw[::2].astype(np.float32)
            Q = raw[1::2].astype(np.float32)
            iq_complex = torch.complex(torch.from_numpy(I), torch.from_numpy(Q))
            iq_complex = iq_complex / 32767.0
            iq_complex = iq_complex.to(DEVICE)
            
            num_slices = len(iq_complex) // SAMPLES_PER_IMAGE
            saved_count = 0 
            
            for i in range(num_slices):
                if saved_count >= 500: break
                
                chunk = iq_complex[i*SAMPLES_PER_IMAGE : (i+1)*SAMPLES_PER_IMAGE]
                tensor_data = generate_tensor_data(chunk, IMAGE_SIZE)
                
                torch.save(tensor_data.cpu(), os.path.join(save_dir, f"real_{i:04d}.pt"))
                saved_count += 1
                
        except Exception as e:
            print(f"Error {label}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nDone! Dataset saved at {DATASET_ROOT}")