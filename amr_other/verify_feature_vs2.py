import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# ================= 用户配置区域 =================
# 可选: 'BPSK', 'QPSK', '8PSK', 'OQPSK', '16QAM', '64QAM'
TARGET_MOD = '16QAM' 

# 测试的频偏点 (Hz)
FREQ_POINTS = [0, 500, 2000] 
# ===============================================

def generate_signal(mod_type, n_samples=2048, sps=4):
    """简易全类别信号生成器"""
    n_syms = n_samples // sps + 50
    
    if mod_type == 'QPSK':
        idx = np.random.randint(0, 4, n_syms)
        syms = np.exp(1j * (np.pi/4 + idx * np.pi/2))
    elif mod_type == '8PSK':
        idx = np.random.randint(0, 8, n_syms)
        syms = np.exp(1j * idx * 2*np.pi/8)
    elif mod_type == 'BPSK':
        syms = 2*np.random.randint(0, 2, n_syms) - 1 + 0j
    elif mod_type == '16QAM':
        i = 2*np.random.randint(0, 4, n_syms) - 3
        q = 2*np.random.randint(0, 4, n_syms) - 3
        syms = i + 1j*q
    elif mod_type == '64QAM':
        i = 2*np.random.randint(0, 8, n_syms) - 7
        q = 2*np.random.randint(0, 8, n_syms) - 7
        syms = i + 1j*q
    elif mod_type == 'OQPSK':
        # OQPSK 需要特殊处理上采样
        idx = np.random.randint(0, 4, n_syms)
        base = np.exp(1j * (np.pi/4 + idx * np.pi/2))
        i, q = base.real, base.imag
        sig_i = np.zeros(n_syms*sps); sig_i[::sps] = i
        sig_q = np.zeros(n_syms*sps); sig_q[::sps] = q
        sig_q = np.roll(sig_q, sps//2) # Offset
        sig_up = sig_i + 1j*sig_q
        # 跳过通用上采样
        return _apply_filter(sig_up, sps)[:n_samples]
    
    # 通用上采样
    sig_up = np.zeros(n_syms * sps, dtype=complex)
    sig_up[::sps] = syms
    return _apply_filter(sig_up, sps)[:n_samples]

def _apply_filter(sig_up, sps):
    # RRC Filter
    t = np.arange(-4*sps, 4*sps+1) / sps
    t[t==0] = 1e-8
    beta = 0.35
    num = np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta))
    den = np.pi*t*(1-(4*beta*t)**2)
    rrc = num/den
    return scipy.signal.convolve(sig_up, rrc, mode='same')

def add_cfo(sig, freq_hz, fs=100000):
    t = np.arange(len(sig)) / fs
    return sig * np.exp(1j * 2 * np.pi * freq_hz * t)

# --- 特征提取核心 (Ultimate Logic) ---
def fast_hist2d(x, y, bins=64, range_bounds=((-1, 1), (-1, 1))):
    H, _, _ = np.histogram2d(x, y, bins=bins, range=range_bounds)
    return np.log1p(H).T

def extract_features(iq):
    # 归一化
    iq = iq / (np.sqrt(np.mean(np.abs(iq)**2)) + 1e-9)
    mag = np.abs(iq)
    
    # [R] ATM (Mag[t-1] vs Mag[t])
    # 重点观察 QAM 的格子结构是否保持
    img_r = fast_hist2d(mag[:-1], mag[1:], bins=64, range_bounds=[[0, 1.8], [0, 1.8]])
    
    # [A] Mag vs Inst-Freq
    diff_sample = iq[1:] * np.conj(iq[:-1])
    inst_freq = np.angle(diff_sample)
    mag_sample = mag[1:]
    # 重点观察 QPSK 的过零菱形是否保持
    img_a = fast_hist2d(inst_freq, mag_sample, bins=64, range_bounds=[[-1.5, 1.5], [0, 1.8]])
    
    # [D] Downsampled Diff
    # 寻找最佳采样点
    best_offset = 0
    max_e = 0
    for i in range(4):
        e = np.sum(np.abs(iq[i::4])**2)
        if e > max_e: max_e = e; best_offset = i
    iq_down = iq[best_offset::4]
    
    # 再次归一化
    iq_down = iq_down / (np.sqrt(np.mean(np.abs(iq_down)**2)) + 1e-9)
    diff = iq_down[1:] * np.conj(iq_down[:-1])
    # 重点观察 PSK 的点是否聚类
    img_d = fast_hist2d(diff.real, diff.imag, bins=64, range_bounds=[[-1.8, 1.8], [-1.8, 1.8]])
    
    return img_r, img_a, img_d

# --- 运行逻辑 ---
def run():
    print(f"=== Testing Robustness for {TARGET_MOD} ===")
    np.random.seed(999) # 固定种子
    
    sig_ref = generate_signal(TARGET_MOD)
    
    rows = len(FREQ_POINTS)
    cols = 3
    plt.figure(figsize=(12, 3*rows))
    plt.suptitle(f"Feature Stability: {TARGET_MOD}", y=1.02, fontsize=16)
    
    for i, freq in enumerate(FREQ_POINTS):
        print(f"  > Processing {freq} Hz...")
        sig_test = add_cfo(sig_ref, freq)
        r, a, d = extract_features(sig_test)
        
        # Plot R
        ax1 = plt.subplot(rows, cols, i*cols + 1)
        ax1.imshow(r, origin='lower', cmap='inferno')
        if i==0: ax1.set_title("R (ATM)\nExpect: Invariant")
        ax1.set_ylabel(f"CFO={freq}Hz")
        ax1.set_xticks([]); ax1.set_yticks([])
        
        # Plot A
        ax2 = plt.subplot(rows, cols, i*cols + 2)
        ax2.imshow(a, origin='lower', cmap='viridis')
        if i==0: ax2.set_title("A (Mag-Freq)\nExpect: Shift")
        ax2.set_xticks([]); ax2.set_yticks([])
        
        # Plot D
        ax3 = plt.subplot(rows, cols, i*cols + 3)
        ax3.imshow(d, origin='lower', cmap='magma')
        if i==0: ax3.set_title("D (Diff)\nExpect: Rotate")
        ax3.set_xticks([]); ax3.set_yticks([])
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()