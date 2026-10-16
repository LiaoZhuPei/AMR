import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal

# ================= 核心逻辑复用 =================
# 为了脚本独立性，这里简化复刻核心函数
# 1. 信号生成 (QPSK)
def generate_qpsk_signal(n_samples=1024, sps=4):
    n_syms = n_samples // sps + 10
    idx = np.random.randint(0, 4, n_syms)
    syms = np.exp(1j * (np.pi/4 + idx * np.pi/2))
    
    # 上采样 + RRC
    sig_up = np.zeros(n_syms * sps, dtype=complex)
    sig_up[::sps] = syms
    
    # 简易 RRC
    t = np.arange(-4*sps, 4*sps+1) / sps
    t[t==0] = 1e-8
    beta = 0.35
    num = np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta))
    den = np.pi*t*(1-(4*beta*t)**2)
    rrc = num/den
    
    sig = scipy.signal.convolve(sig_up, rrc, mode='same')
    return sig[:n_samples]

# 2. 施加频偏
def add_cfo(sig, freq_hz, fs=100000):
    t = np.arange(len(sig)) / fs
    return sig * np.exp(1j * 2 * np.pi * freq_hz * t)

# 3. 特征提取 (究极体版本)
def fast_hist2d(x, y, bins=64, range_bounds=((-1, 1), (-1, 1))):
    H, _, _ = np.histogram2d(x, y, bins=bins, range=range_bounds)
    return np.log1p(H).T

def extract_rad_features(iq):
    # 归一化
    iq = iq / (np.sqrt(np.mean(np.abs(iq)**2)) + 1e-9)
    mag = np.abs(iq)
    
    # [R] ATM: Mag[t-1] vs Mag[t]
    # 理论上绝对抗频偏
    img_r = fast_hist2d(mag[:-1], mag[1:], bins=64, range_bounds=[[0, 1.5], [0, 1.5]])
    
    # [A] Mag vs Inst-Freq
    # 理论上频偏会导致 Inst-Freq 平移 (X轴移动)
    diff_sample = iq[1:] * np.conj(iq[:-1])
    inst_freq = np.angle(diff_sample)
    mag_sample = mag[1:]
    img_a = fast_hist2d(inst_freq, mag_sample, bins=64, range_bounds=[[-1.5, 1.5], [0, 1.5]])
    
    # [D] Downsampled Diff (SPS=4)
    # 理论上频偏会导致整体旋转
    best_offset = 0
    max_e = 0
    for i in range(4):
        e = np.sum(np.abs(iq[i::4])**2)
        if e > max_e: max_e = e; best_offset = i
    iq_down = iq[best_offset::4]
    
    # 重新归一化
    iq_down = iq_down / (np.sqrt(np.mean(np.abs(iq_down)**2)) + 1e-9)
    diff = iq_down[1:] * np.conj(iq_down[:-1])
    img_d = fast_hist2d(diff.real, diff.imag, bins=64, range_bounds=[[-1.5, 1.5], [-1.5, 1.5]])
    
    return img_r, img_a, img_d

# ================= 主实验 =================
def run_stress_test():
    print("Generating Reference Signal...")
    # 固定种子，保证波形一样，只受频偏影响
    np.random.seed(42) 
    sig_ref = generate_qpsk_signal()
    
    # 设定扫描点：0Hz (基准), 500Hz (中等), 2000Hz (严重, 20ppm@2.4G)
    freq_points = [0, 500, 2000,5000] 
    
    plt.figure(figsize=(15, 10))
    plt.suptitle("Feature Robustness Stress Test (QPSK)", fontsize=16)
    
    rows = len(freq_points)
    cols = 3 # R, A, D
    
    for i, freq in enumerate(freq_points):
        print(f"Testing Freq Offset: {freq} Hz...")
        
        # 施加频偏
        sig_test = add_cfo(sig_ref, freq)
        
        # 提取特征
        img_r, img_a, img_d = extract_rad_features(sig_test)
        
        # 绘图
        # R 通道
        plt.subplot(rows, cols, i*cols + 1)
        plt.imshow(img_r, origin='lower', cmap='inferno')
        plt.title(f"R (ATM) @ {freq}Hz\n(Should be Invariant)")
        plt.axis('off')
        
        # A 通道
        plt.subplot(rows, cols, i*cols + 2)
        plt.imshow(img_a, origin='lower', cmap='viridis')
        plt.title(f"A (Mag-Freq) @ {freq}Hz\n(Should Shift X)")
        plt.axis('off')
        
        # D 通道
        plt.subplot(rows, cols, i*cols + 3)
        plt.imshow(img_d, origin='lower', cmap='magma')
        plt.title(f"D (Diff) @ {freq}Hz\n(Should Rotate)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    print("Done. Check the plot.")

if __name__ == "__main__":
    run_stress_test()