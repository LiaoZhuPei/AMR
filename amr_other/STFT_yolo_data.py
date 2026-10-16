import os
import shutil
import numpy as np
import scipy.signal
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib

# ================= 配置区域 =================
# [修改] 数据集路径，名字改为 Spectrogram 以作区分
DATASET_ROOT = "./dataset_Spectrogram_Image_VSG60"
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
# [建议] 时频图建议分辨率调高一点以保留频段细节，不过YOLO内部默认通常是128或224
# 如果显存较小可以继续用 64，这里暂时设为 64 保持与之前兼容
IMAGE_SIZE = 64 
TOTAL_SIM_SAMPLES_PER_CLASS = 1000
TRAIN_VAL_RATIO = 0.8
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ================= 1. VSG 仿真器 (全物理特性版) - 保持不变 =================
class LabDeviceSimulator:
    def __init__(self, sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9):
        self.fs = sample_rate
        self.rs = symbol_rate
        self.fc = carrier_freq
        self.sps = int(sample_rate / symbol_rate)
        self.default_rrc = self._generate_rrc(0.35, 10, self.sps)

    def _generate_rrc(self, alpha, span, sps):
        t = np.arange(-span*sps, span*sps+1) / sps
        t[t == 0] = 1e-8
        t[np.abs(t) == 1/(4*alpha)] += 1e-8
        num = np.sin(np.pi*t*(1-alpha)) + 4*alpha*t*np.cos(np.pi*t*(1+alpha))
        den = np.pi*t*(1-(4*alpha*t)**2)
        return num / den

    def generate_packet(self, mod_type, num_samples, ppm_error, iq_gain_db, iq_phase_deg, snr_db):
        num_symbols = int(num_samples / self.sps * 1.2) + 50
        syms = self._modulate(mod_type, num_symbols)
        
        sig_base = np.zeros(num_symbols * self.sps, dtype=complex)
        if mod_type == 'OQPSK':
            i, q = syms.real, syms.imag
            sig_i = np.zeros_like(sig_base, dtype=float); sig_i[::self.sps] = i
            sig_q = np.zeros_like(sig_base, dtype=float); sig_q[::self.sps] = q
            sig_q = np.roll(sig_q, self.sps // 2)
            sig_base = sig_i + 1j * sig_q
        else:
            sig_base[::self.sps] = syms

        curr_alpha = np.random.uniform(0.2, 0.6)
        curr_taps = self._generate_rrc(curr_alpha, 10, self.sps)
        sig = scipy.signal.convolve(sig_base, curr_taps, mode='same')

        sro_factor = 1.0 + (ppm_error * 1e-6)
        new_len = int(len(sig) / sro_factor)
        sig = scipy.signal.resample(sig, new_len) 

        dac_res = 2**14
        peak = np.max(np.abs(sig)) + 1e-9
        sig = sig / peak
        sig = np.round(sig * (dac_res/2)) / (dac_res/2)

        sig = self._apply_iq_imbalance(sig, iq_gain_db, iq_phase_deg)
        
        dc_mag = 10**(-35/20) 
        dc_phase = np.random.uniform(0, 2*np.pi)
        sig = sig + dc_mag * np.exp(1j * dc_phase)

        cfo_hz = self.fc * ppm_error * 1e-6
        t = np.arange(len(sig)) / self.fs
        rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
        
        phase_jitter = np.cumsum(np.random.normal(0, 0.05 * np.pi/180, len(sig)))
        sig = sig * rotation * np.exp(1j * phase_jitter)

        pwr = np.mean(np.abs(sig)**2)
        n_pwr = pwr / (10**(snr_db/10.0))
        noise = (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig))) * np.sqrt(n_pwr/2)
        sig = sig + noise
        
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

# ================= 2. [核心替换] 时频图生成模块 =================

# 预先加载 colormap 提速，避免内存泄漏
try:
    cmap = matplotlib.colormaps['viridis']
except AttributeError:
    import matplotlib.cm as cm
    cmap = cm.viridis

def generate_spectrogram_image(iq_complex_data, img_size=64):
    """
    将一维的复数IQ信号通过 STFT(短时傅里叶变换) 转换为 2D 时频图
    并映射为 RGB 图像 (伪彩色)，以便被 YOLO / CNN 识别。
    """
    # 转换为 numpy
    if torch.is_tensor(iq_complex_data):
        iq_data = iq_complex_data.cpu().numpy().squeeze()
    else:
        iq_data = iq_complex_data

    # 当片段极短时的安全保护
    if len(iq_data) < 256: 
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # 1. 计算短时傅里叶变换 (STFT)
    # nperseg(窗长) 决定频率分辨率, noverlap 决定时间平滑度
    # 对于 10240 个点，设置 nperseg=256, noverlap=128 会产生 (256, 79) 大小的二维矩阵
    f, t, Sxx = scipy.signal.spectrogram(
        iq_data, 
        fs=1.0, 
        window='hann', 
        nperseg=256, 
        noverlap=128, 
        return_onesided=False, 
        mode='magnitude'
    )
    
    # 2. 零频点居中 (非常关键，将复数信号的负频率与正频率正确对齐)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    
    # 3. 对数尺度化 (转为 dB)
    Sxx_db = 20 * np.log10(Sxx + 1e-12)
    
    # 4. 归一化到 [0, 1] 范围
    s_min = np.min(Sxx_db)
    s_max = np.max(Sxx_db)
    if s_max > s_min:
        Sxx_norm = (Sxx_db - s_min) / (s_max - s_min)
    else:
        Sxx_norm = np.zeros_like(Sxx_db)
        
    # 5. 伪彩色映射 (使用 Viridis，返回 RGBA)
    # 截取前三个通道 RGB，缩放到[0, 255]
    img_rgb = (cmap(Sxx_norm)[:, :, :3] * 255).astype(np.uint8)
    
    # 6. 使用 PIL 进行插值调整到目标网络输入尺寸 (IMAGE_SIZE x IMAGE_SIZE)
    img = Image.fromarray(img_rgb)
    img = img.resize((img_size, img_size), Image.BILINEAR)
    
    # 7. 坐标轴翻转：
    # fftshift 后，默认矩阵中第0行是最低频率(-Fs/2)，底部是最高频率(Fs/2)。
    # 通常的频谱图习惯上高频在图像上方，因此我们把图像进行一次垂直翻转。
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    return np.array(img)

# ================= 执行生成 =================
if __name__ == "__main__":

    if os.path.exists(DATASET_ROOT):
        shutil.rmtree(DATASET_ROOT)
    
    print("===== Generating Spectrogram Image Data (.jpg) =====")
    print("Features: STFT (Short-Time Fourier Transform) Spectrograms")
    print("Colormap: Viridis | Output Size:", IMAGE_SIZE, "x", IMAGE_SIZE)
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
            
            # 仿真参数注入
            snr = np.random.uniform(10, 25)
            ppm = np.random.uniform(-40, 40) 
            gain = np.random.uniform(-0.5, 0.5)
            phase = np.random.uniform(-2, 2)
            
            # 生成带损伤复数信号
            sig = sim.generate_packet(cls_name, SAMPLES_PER_IMAGE, ppm, gain, phase, snr)
            
            #[核心替换] 调用时频图生成函数
            img_arr = generate_spectrogram_image(sig, IMAGE_SIZE)
            
            # 保存为 jpg 图片
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
            iq_complex = I + 1j * Q
            iq_complex = iq_complex / 32767.0 # 归一化
            
            num_slices = len(iq_complex) // SAMPLES_PER_IMAGE
            saved_count = 0 
            
            for i in range(num_slices):
                if saved_count >= 500: break
                
                chunk = iq_complex[i*SAMPLES_PER_IMAGE : (i+1)*SAMPLES_PER_IMAGE]
                
                # [核心替换] 调用时频图生成函数
                img_arr = generate_spectrogram_image(chunk, IMAGE_SIZE)
                
                save_path = os.path.join(save_dir, f"real_{i:04d}.jpg")
                Image.fromarray(img_arr).save(save_path)
                
                saved_count += 1
                
        except Exception as e:
            print(f"Error {label}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nDone! Dataset saved at {DATASET_ROOT}")