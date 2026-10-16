import os
import shutil
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# ================= 配置区域 =================
# [修改] 数据集路径，名字改为Image以作区分
DATASET_ROOT = "./dataset_tri_RAD_Image_VSG60"
SIM_TRAIN_DIR = os.path.join(DATASET_ROOT, "sim_train")
SIM_VAL_DIR = os.path.join(DATASET_ROOT, "sim_val")
REAL_TEST_DIR = os.path.join(DATASET_ROOT, "real_test")

CLASS_NAMES =[
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
    "E:/Project/AMR-Benchmark/OQPSK-03-24-25-10h54m02s6.iq": "OQPSK",
    "E:/Project/AMR-Benchmark/PI4DQPSK-03-24-25-10h54m23s982.iq": "PI4DQPSK"
}

SAMPLES_PER_IMAGE = 10240  
IMAGE_SIZE = 224
TOTAL_SIM_SAMPLES_PER_CLASS = 1000
TRAIN_VAL_RATIO = 0.8
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ================= 1. VSG 仿真器 (全物理特性版) =================
class LabDeviceSimulator:
    def __init__(self, sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9):
        self.fs = sample_rate
        self.rs = symbol_rate
        self.fc = carrier_freq
        self.sps = int(sample_rate / symbol_rate)
        # 默认滤波器
        self.default_rrc = self._generate_rrc(0.35, 10, self.sps)

    def _generate_rrc(self, alpha, span, sps):
        t = np.arange(-span*sps, span*sps+1) / sps
        t[t == 0] = 1e-8
        t[np.abs(t) == 1/(4*alpha)] += 1e-8
        num = np.sin(np.pi*t*(1-alpha)) + 4*alpha*t*np.cos(np.pi*t*(1+alpha))
        den = np.pi*t*(1-(4*alpha*t)**2)
        return num / den

    def generate_packet(self, mod_type, num_samples, ppm_error, iq_gain_db, iq_phase_deg, snr_db):
        """
        生成全流程物理仿真信号：
        Mod -> Upsample -> ISI(Random Filter) -> SRO(Resample) -> DAC(Quantization) 
        -> IQ Imbalance -> DC Offset -> CFO -> Phase Noise -> AWGN
        """
        # 1. 基础调制
        # 稍微多生成一点符号，防止重采样后长度不够
        num_symbols = int(num_samples / self.sps * 1.2) + 50
        syms = self._modulate(mod_type, num_symbols)
        
        # 2. 上采样
        sig_base = np.zeros(num_symbols * self.sps, dtype=complex)
        if mod_type == 'OQPSK':
            i, q = syms.real, syms.imag
            sig_i = np.zeros_like(sig_base, dtype=float); sig_i[::self.sps] = i
            sig_q = np.zeros_like(sig_base, dtype=float); sig_q[::self.sps] = q
            sig_q = np.roll(sig_q, self.sps // 2)
            sig_base = sig_i + 1j * sig_q
        else:
            sig_base[::self.sps] = syms

        # 3. [关键] 随机化成形滤波器 (模拟 ISI / 滤波器失配)
        # alpha 在 0.2 ~ 0.6 之间波动，模拟不同设备的带宽限制差异
        curr_alpha = np.random.uniform(0.2, 0.6)
        curr_taps = self._generate_rrc(curr_alpha, 10, self.sps)
        sig = scipy.signal.convolve(sig_base, curr_taps, mode='same')

        # 4. [关键] 采样率误差 (SRO)
        # 模拟晶振不仅影响频偏，也影响采样率
        # SRO = ppm * 1e-6
        sro_factor = 1.0 + (ppm_error * 1e-6)
        new_len = int(len(sig) / sro_factor)
        sig = scipy.signal.resample(sig, new_len) # 引入插值模糊

        # 5.[关键] 14-bit DAC 量化
        dac_res = 2**14
        # 归一化到 +/- 1
        peak = np.max(np.abs(sig)) + 1e-9
        sig = sig / peak
        sig = np.round(sig * (dac_res/2)) / (dac_res/2)

        # 6. 硬件损伤 (IQ Imbalance)
        sig = self._apply_iq_imbalance(sig, iq_gain_db, iq_phase_deg)
        
        # 7. [关键] 直流偏置 (DC Offset)
        # 模拟本振泄露，通常在 -40dBc 到 -30dBc
        dc_mag = 10**(-35/20) 
        dc_phase = np.random.uniform(0, 2*np.pi)
        sig = sig + dc_mag * np.exp(1j * dc_phase)

        # 8. 载波频偏 (CFO)
        cfo_hz = self.fc * ppm_error * 1e-6
        t = np.arange(len(sig)) / self.fs
        rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
        
        # 9. [关键] 相位噪声 (Phase Noise)
        # 模拟本振抖动，累积随机相位
        phase_jitter = np.cumsum(np.random.normal(0, 0.05 * np.pi/180, len(sig)))
        sig = sig * rotation * np.exp(1j * phase_jitter)

        # 10. 热噪声 (AWGN)
        pwr = np.mean(np.abs(sig)**2)
        n_pwr = pwr / (10**(snr_db/10.0))
        noise = (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig))) * np.sqrt(n_pwr/2)
        sig = sig + noise
        
        # 截断/补齐到目标长度
        if len(sig) >= num_samples:
            return sig[:num_samples]
        else:
            return np.concatenate([sig, np.zeros(num_samples - len(sig))])

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

# ================= 2. 究极体特征提取逻辑 =================

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
    Ch1 (R): ATM (Amplitude Transition Matrix) -> 抗频偏，分QAM
    Ch2 (A): Mag-Freq (InstFreq vs Mag) -> 抗平移，分QPSK/OQPSK
    Ch3 (D): Downsampled Diff (Best Offset) -> 抗SRO，分PSK
    """
    if iq_complex_data.dim() == 1:
        iq_complex_data = iq_complex_data.unsqueeze(0)
    iq_data = iq_complex_data.squeeze()
    
    if iq_data.numel() < 100: 
        return torch.zeros(3, img_size, img_size)

    # 1. 归一化
    scale = torch.sqrt(torch.mean(iq_data.real**2 + iq_data.imag**2)) + 1e-9
    norm_iq = iq_data / (3.0 * scale)
    mag = torch.abs(norm_iq)
    
    # ----------------------------------------------------
    # [Channel 1 - R] 幅度转移矩阵 (ATM)
    # ----------------------------------------------------
    # mag[t-1] vs mag[t]
    mag_curr = mag[1:]
    mag_prev = mag[:-1]
    hist_r = fast_hist2d(mag_prev, mag_curr, bins=img_size, range_bounds=((-0.1, 1.5), (-0.1, 1.5)))
    
    # ----------------------------------------------------
    # [Channel 2 - A] 幅度 vs 瞬时频率 (Mag-Freq)
    # ----------------------------------------------------
    diff_sample = norm_iq[1:] * torch.conj(norm_iq[:-1])
    inst_freq = torch.angle(diff_sample) # [-pi, pi]
    mag_sample = mag[1:]
    
    # X: InstFreq, Y: Mag
    hist_a = fast_hist2d(inst_freq, mag_sample, bins=img_size, range_bounds=((-1.5, 1.5), (0.0, 1.5)))
    
    # ----------------------------------------------------
    # [Channel 3 - D] 降采样差分相位 (带最佳采样点搜索)
    # ----------------------------------------------------
    best_offset = 0
    max_energy = -1.0
    # 遍历 4 个偏移量 (假设 SPS=4)
    for offset in range(4):
        temp_down = norm_iq[offset::4]
        en = torch.sum(torch.abs(temp_down)**2)
        if en > max_energy:
            max_energy = en
            best_offset = offset
            
    iq_down = norm_iq[best_offset::4]
    
    # 降采样后重新归一化 (关键！)
    pwr = torch.sqrt(torch.mean(iq_down.real**2 + iq_down.imag**2)) + 1e-9
    iq_down = iq_down / pwr
    
    # 计算差分
    diff_signal = iq_down[1:] * torch.conj(iq_down[:-1])
    
    # 差分后再次归一化 (防止越界)
    d_scale = torch.sqrt(torch.mean(diff_signal.real**2 + diff_signal.imag**2)) + 1e-9
    diff_signal = diff_signal / (3.0 * d_scale)
    
    d_real = diff_signal.real
    d_imag = diff_signal.imag
    
    hist_d = fast_hist2d(d_real, d_imag, bins=img_size, range_bounds=((-1.5, 1.5), (-1.5, 1.5)))
    
    # ----------------------------------------------------
    # 堆叠
    # ----------------------------------------------------
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
    
    print("===== Generating VSG-Simulated Image Data (.jpg) =====")
    print("Features: ATM, Mag-Freq, D-DownDiff")
    print("Physics: SRO, PhaseNoise, ISI, DC, Quantization")
    
    # 初始化仿真器
    sim = LabDeviceSimulator(sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9)
    
    # 1. 仿真数据生成
    for cls_name in CLASS_NAMES:
        os.makedirs(os.path.join(SIM_TRAIN_DIR, cls_name), exist_ok=True)
        os.makedirs(os.path.join(SIM_VAL_DIR, cls_name), exist_ok=True)
        
        print(f"Generating Sim {cls_name}...")
        for i in tqdm(range(TOTAL_SIM_SAMPLES_PER_CLASS)):
            target_dir = SIM_TRAIN_DIR if np.random.rand() < TRAIN_VAL_RATIO else SIM_VAL_DIR
            target_dir = os.path.join(target_dir, cls_name)
            
            # [参数范围] 加大频偏，覆盖真实场景
            snr = np.random.uniform(10, 25)
            ppm = np.random.uniform(-40, 40) # 96kHz 频偏
            gain = np.random.uniform(-0.5, 0.5)
            phase = np.random.uniform(-2, 2)
            
            # 生成信号
            sig = sim.generate_packet(cls_name, SAMPLES_PER_IMAGE, ppm, gain, phase, snr)
            
            if isinstance(sig, np.ndarray): 
                sig_tensor = torch.from_numpy(sig).to(DEVICE)
            
            # 得到 (3, 64, 64) 范围在 [0, 1] 的张量
            tensor_data = generate_tensor_data(sig_tensor, IMAGE_SIZE)
            
            # [修改点] 将张量维度重排为 (64, 64, 3) 并映射到 [0, 255]，转换为图像保存
            img_arr = (tensor_data.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            save_path = os.path.join(target_dir, f"{cls_name}_{i:04d}_SNR{snr:.1f}.jpg")
            Image.fromarray(img_arr).save(save_path)

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
                
                # [修改点] 真实数据同样转为 jpg 保存
                img_arr = (tensor_data.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                save_path = os.path.join(save_dir, f"real_{i:04d}.jpg")
                Image.fromarray(img_arr).save(save_path)
                
                saved_count += 1
                
        except Exception as e:
            print(f"Error {label}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nDone! Dataset saved at {DATASET_ROOT}")