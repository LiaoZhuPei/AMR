import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# ================= 1. 仿真器定义 (保持之前的高保真逻辑) =================
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

    def generate_packet(self, mod_type, num_symbols, ppm_error, iq_gain_db, iq_phase_deg, snr_db):
        # 1. 调制
        syms = self._modulate(mod_type, num_symbols)
        
        # 2. 上采样 (含 OQPSK 处理)
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

        # 3. IQ 不平衡
        sig_impaired = self._apply_iq_imbalance(sig_base, iq_gain_db, iq_phase_deg)

        # 4. 成形滤波
        sig_shaped = scipy.signal.convolve(sig_impaired, self.rrc_taps, mode='same')

        # 5. 频偏 (关键！)
        cfo_hz = self.fc * ppm_error * 1e-6
        t = np.arange(len(sig_shaped)) / self.fs
        rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
        sig_rotated = sig_shaped * rotation

        # 6. 噪声
        sig_power = np.mean(np.abs(sig_rotated)**2)
        noise_power = sig_power / (10**(snr_db/10.0))
        noise = (np.random.randn(len(sig_rotated)) + 1j*np.random.randn(len(sig_rotated))) * np.sqrt(noise_power/2)
        
        return sig_rotated + noise

    def _modulate(self, mod_type, n):
        if mod_type == 'QPSK' or mod_type == 'OQPSK':
            return np.exp(1j * (np.pi/4 + np.random.randint(0, 4, n) * np.pi/2))
        elif mod_type == '8PSK':
            return np.exp(1j * np.random.randint(0, 8, n) * 2*np.pi/8)
        return np.random.randn(n) + 1j*np.random.randn(n)

    def _apply_iq_imbalance(self, sig, gain_db, phase_deg):
        gain = 10**(gain_db / 20.0)
        phase_rad = phase_deg * np.pi / 180.0
        i_new = sig.real
        q_new = gain * (sig.real * np.sin(phase_rad) + sig.imag * np.cos(phase_rad))
        return i_new + 1j * q_new

# ================= 2. 实验：对比 QPSK vs OQPSK 的 R/A/D 特征 =================

def run_feature_check():
    sim = LabDeviceSimulator(carrier_freq=2.4e9)
    
    # 设定损伤参数 (模拟真实的不完美设备)
    # 5 ppm 频偏 (会导致 R 通道变成圆环)
    # 20dB SNR
    PPM = 5
    SNR = 10
    
    # 生成两个信号
    sig_qpsk = sim.generate_packet('QPSK', 1024, ppm_error=PPM, iq_gain_db=0.5, iq_phase_deg=1.0, snr_db=SNR)
    sig_oqpsk = sim.generate_packet('OQPSK', 1024, ppm_error=PPM, iq_gain_db=0.5, iq_phase_deg=1.0, snr_db=SNR)
    
    # --- 提取特征 ---
    # 降采样 (显露 D 通道特征的关键)
    step = 4
    qpsk_down = sig_qpsk[::step]
    oqpsk_down = sig_oqpsk[::step]
    
    # A 通道 (幅度)
    amp_q = np.abs(qpsk_down)
    amp_o = np.abs(oqpsk_down)
    
    # D 通道 (差分)
    diff_q = qpsk_down[1:] * np.conj(qpsk_down[:-1])
    diff_o = oqpsk_down[1:] * np.conj(oqpsk_down[:-1])
    
    # --- 绘图 ---
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Simulated Features (PPM={PPM}, SNR={SNR}dB)", fontsize=16)
    
    # 1. R 通道 (星座图)
    plt.subplot(2, 3, 1)
    plt.scatter(sig_qpsk.real[:2000], sig_qpsk.imag[:2000], s=1, alpha=0.5)
    plt.title("QPSK - R Channel (Ring due to CFO)")
    plt.axis('equal')
    
    plt.subplot(2, 3, 4)
    plt.scatter(sig_oqpsk.real[:2000], sig_oqpsk.imag[:2000], s=1, alpha=0.5, color='orange')
    plt.title("OQPSK - R Channel (Ring due to CFO)")
    plt.axis('equal')
    
    # 2. A 通道 (幅度直方图)
    plt.subplot(2, 3, 2)
    plt.hist(amp_q, bins=100, density=True, range=(0, 2.0))
    plt.title("QPSK - A Channel (Has Zero-Crossing?)")
    # 重点看左边有没有尾巴
    
    plt.subplot(2, 3, 5)
    plt.hist(amp_o, bins=100, density=True, color='orange', range=(0, 2.0))
    plt.title("OQPSK - A Channel (No Zero-Crossing)")
    
    # 3. D 通道 (差分散点图)
    plt.subplot(2, 3, 3)
    plt.scatter(diff_q.real[:2000], diff_q.imag[:2000], s=1, alpha=0.5)
    plt.title("QPSK - D Channel (4 Clouds?)")
    plt.axis('equal')
    
    plt.subplot(2, 3, 6)
    plt.scatter(diff_o.real[:2000], diff_o.imag[:2000], s=1, alpha=0.5, color='orange')
    plt.title("OQPSK - D Channel (Ring?)")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_feature_check()