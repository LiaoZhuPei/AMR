import os
import shutil
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ================= 实验配置矩阵 =================
# 将依次生成这 4 组数据
EXPERIMENTS = [
    {'name': 'Exp1_Old_Clean',   'use_new_feat': False, 'add_cfo': False},
    {'name': 'Exp2_New_Clean',   'use_new_feat': True,  'add_cfo': False},
    {'name': 'Exp3_Old_Rotated', 'use_new_feat': False, 'add_cfo': True},
    {'name': 'Exp4_New_Rotated', 'use_new_feat': True,  'add_cfo': True},
]

ROOT_DIR = "./dataset_Ablation"
CLASS_NAMES = [
    "BPSK", "QPSK", "8PSK", "OQPSK", "PI4DQPSK", 
    "16QAM", "32QAM", "64QAM", 
    "2ASK", "16APSK", "32APSK", "FSK2"
]

SAMPLES_PER_CLASS = 1000  # 每个类别的样本数
SAMPLES_LEN = 5120       # 信号长度
IMG_SIZE = 64
DEVICE = torch.device("cpu") # 生成数据用 CPU 即可，防止显存溢出

# ================= 1. VSG 仿真器 (含完整 OQPSK) =================
class LabDeviceSimulator:
    def __init__(self, sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9):
        self.fs = sample_rate
        self.sps = int(sample_rate / symbol_rate)
        self.fc = carrier_freq
        # 生成 RRC 滤波器
        self.rrc_taps = self._gen_rrc(0.35, 10, self.sps)

    def _gen_rrc(self, alpha, span, sps):
        t = np.arange(-span*sps, span*sps+1) / sps
        t[t == 0] = 1e-8
        t[np.abs(t) == 1/(4*alpha)] += 1e-8
        num = np.sin(np.pi*t*(1-alpha)) + 4*alpha*t*np.cos(np.pi*t*(1+alpha))
        den = np.pi*t*(1-(4*alpha*t)**2)
        return num / den

    def generate(self, mod_type, n_samples, ppm_error, snr_db, seed):
        """
        生成仿真信号
        seed: 随机种子，用于锁定波形内容，确保 Clean 和 Rotated 只有频偏不同
        """
        np.random.seed(seed)
        
        # 1. 计算需要的符号数
        n_syms = n_samples // self.sps + 50
        
        # 2. 调制与上采样 (含 OQPSK 逻辑)
        sig_base = self._modulate_and_upsample(mod_type, n_syms)
        
        # 3. 成形滤波
        sig = scipy.signal.convolve(sig_base, self.rrc_taps, mode='same')
        sig = sig[:n_samples] # 截断到指定长度
        
        # 4. 频偏 (由参数控制)
        if ppm_error != 0:
            cfo_hz = self.fc * ppm_error * 1e-6
            t = np.arange(len(sig)) / self.fs
            sig = sig * np.exp(1j * 2 * np.pi * cfo_hz * t)
            
        # 5. 噪声 (受 seed 控制，保证噪声形态一致)
        pwr = np.mean(np.abs(sig)**2)
        noise_pwr = pwr / (10**(snr_db/10.0))
        noise = (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig))) * np.sqrt(noise_pwr/2)
        
        return sig + noise

    def _modulate_and_upsample(self, mod, n):
        # 通用上采样容器
        sig_up = np.zeros(n * self.sps, dtype=complex)
        
        if mod == 'OQPSK':
            # [关键] OQPSK 完整逻辑
            idx = np.random.randint(0, 4, n)
            base = np.exp(1j * (np.pi/4 + idx * np.pi/2))
            i, q = base.real, base.imag
            
            sig_i = np.zeros(n * self.sps)
            sig_q = np.zeros(n * self.sps)
            
            sig_i[::self.sps] = i
            sig_q[::self.sps] = q
            
            # Q 路延迟半个符号 (Offset)
            sig_q = np.roll(sig_q, self.sps // 2)
            
            # 组合
            return sig_i + 1j * sig_q

        elif mod == 'QPSK':
            idx = np.random.randint(0, 4, n)
            syms = np.exp(1j * (np.pi/4 + idx * np.pi/2))
        elif mod == '8PSK':
            idx = np.random.randint(0, 8, n)
            syms = np.exp(1j * idx * 2*np.pi/8)
        elif mod == 'BPSK':
            syms = 2*np.random.randint(0, 2, n) - 1 + 0j
        elif mod == '16QAM':
            i = 2*np.random.randint(0, 4, n) - 3
            q = 2*np.random.randint(0, 4, n) - 3
            syms = i + 1j*q
        elif mod == '32QAM':
            i = np.random.randint(-5, 6, n); q = np.random.randint(-5, 6, n)
            syms = i + 1j*q
        elif mod == '64QAM':
            i = 2*np.random.randint(0, 8, n) - 7; q = 2*np.random.randint(0, 8, n) - 7
            syms = i + 1j*q
        elif mod == 'PI4DQPSK':
            bits = np.random.randint(0, 4, n)
            shifts = np.array([np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4])[bits]
            phases = np.cumsum(shifts)
            syms = np.exp(1j * phases)
        elif mod == '2ASK':
            syms = np.random.randint(0, 2, n).astype(float) + 0j
        elif mod == '16APSK':
            r = np.random.choice([1.0, 2.5], n)
            th = np.random.uniform(0, 2*np.pi, n)
            syms = r * np.exp(1j*th)
        elif mod == '32APSK':
            r = np.random.choice([1.0, 2.5, 4.0], n)
            th = np.random.uniform(0, 2*np.pi, n)
            syms = r * np.exp(1j*th)
        elif mod == 'FSK2':
            f_dev = 0.05 
            bits = np.random.randint(0, 2, n)
            freqs = (bits * 2 - 1) * f_dev
            phase = 2 * np.pi * np.cumsum(freqs)
            # FSK 也是连续相位，直接生成即可，不需要上采样插值
            # 这里简单返回过采样的点
            return np.exp(1j * np.repeat(phase, self.sps))
        else:
            syms = np.random.randn(n) + 1j*np.random.randn(n)
            
        # 标准上采样
        sig_up[::self.sps] = syms
        return sig_up

# ================= 2. 特征提取 (Old vs New) =================

def fast_hist2d(x, y, bins=64, range_bounds=None):
    x = torch.clamp(x, range_bounds[0][0], range_bounds[0][1])
    y = torch.clamp(y, range_bounds[1][0], range_bounds[1][1])
    x_idx = ((x - range_bounds[0][0]) / (range_bounds[0][1] - range_bounds[0][0]) * (bins - 1)).long()
    y_idx = ((y - range_bounds[1][0]) / (range_bounds[1][1] - range_bounds[1][0]) * (bins - 1)).long()
    flat_idx = y_idx * bins + x_idx
    hist = torch.bincount(flat_idx, minlength=bins*bins).float()
    return torch.log1p(hist).view(bins, bins)

def extract_features_OLD(sig_np):
    """
    旧特征 (Vulnerable):
    R: Constellation (I vs Q) -> 怕频偏
    A: Phase vs Mag -> 怕频偏
    D: Sample Diff -> 怕过采样
    """
    sig = torch.from_numpy(sig_np)
    sig = sig / (torch.sqrt(torch.mean(torch.abs(sig)**2)) + 1e-9)
    
    # R: Constellation
    r = fast_hist2d(sig.real, sig.imag, bins=64, range_bounds=[[-1.5, 1.5], [-1.5, 1.5]])
    
    # A: Phase vs Mag
    mag = torch.abs(sig)
    phase = torch.angle(sig) / np.pi
    a = fast_hist2d(phase, mag, bins=64, range_bounds=[[-1, 1], [0, 1.8]])
    
    # D: Sample Diff (No Downsample)
    diff = sig[1:] * torch.conj(sig[:-1])
    d = fast_hist2d(diff.real, diff.imag, bins=64, range_bounds=[[-1.5, 1.5], [-1.5, 1.5]])
    
    return torch.stack([r, a, d])

def extract_features_NEW(sig_np):
    """
    新特征 (Robust):
    R: ATM (Mag_prev vs Mag_curr) -> 抗频偏
    A: Mag-Freq (InstFreq vs Mag) -> 抗平移
    D: Downsampled Diff -> 抗过采样
    """
    sig = torch.from_numpy(sig_np)
    sig = sig / (torch.sqrt(torch.mean(torch.abs(sig)**2)) + 1e-9)
    mag = torch.abs(sig)
    
    # R: ATM
    r = fast_hist2d(mag[:-1], mag[1:], bins=64, range_bounds=[[0, 1.8], [0, 1.8]])
    
    # A: Mag-Freq
    diff_sample = sig[1:] * torch.conj(sig[:-1])
    inst_freq = torch.angle(diff_sample)
    a = fast_hist2d(inst_freq, mag[1:], bins=64, range_bounds=[[-1.5, 1.5], [0, 1.8]])
    
    # D: Downsampled Diff
    best_off = 0
    max_e = 0
    for i in range(4):
        e = torch.sum(torch.abs(sig[i::4])**2)
        if e > max_e: max_e=e; best_off=i
    sig_down = sig[best_off::4]
    # Re-normalize
    sig_down = sig_down / (torch.sqrt(torch.mean(torch.abs(sig_down)**2)) + 1e-9)
    diff_down = sig_down[1:] * torch.conj(sig_down[:-1])
    d = fast_hist2d(diff_down.real, diff_down.imag, bins=64, range_bounds=[[-1.8, 1.8], [-1.8, 1.8]])
    
    return torch.stack([r, a, d])

# ================= 3. 主程序 =================
if __name__ == "__main__":
    
    sim = LabDeviceSimulator()
    
    # 依次执行 4 个实验的数据生成
    for exp_config in EXPERIMENTS:
        exp_name = exp_config['name']
        use_new = exp_config['use_new_feat']
        add_cfo = exp_config['add_cfo']
        
        print(f"\n========== Generating {exp_name} ==========")
        print(f"Features: {'NEW (Robust)' if use_new else 'OLD (Vulnerable)'}")
        print(f"CFO Injection: {add_cfo}")
        
        # 路径准备
        exp_dir = os.path.join(ROOT_DIR, exp_name)
        train_dir = os.path.join(exp_dir, "train")
        val_dir = os.path.join(exp_dir, "val")
        
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            
            # 生成样本
            for i in tqdm(range(SAMPLES_PER_CLASS), desc=cls, leave=False):
                # 随机参数
                snr = np.random.uniform(5, 25)
                # 频偏设定: Clean=0, Rotated=Random
                ppm = np.random.uniform(-5, 5) if add_cfo else 0
                
                # 生成波形 (Seed=i 保证波形内容一致)
                sig = sim.generate(cls, SAMPLES_LEN, ppm, snr, seed=i)
                
                # 提取特征
                if use_new:
                    tensor = extract_features_NEW(sig)
                else:
                    tensor = extract_features_OLD(sig)
                
                # 保存
                save_dir = train_dir if i < SAMPLES_PER_CLASS * 0.8 else val_dir
                torch.save(tensor, os.path.join(save_dir, cls, f"{cls}_{i:04d}.pt"))
                
    print("\nAll datasets generated successfully!")
    print(f"Location: {ROOT_DIR}")