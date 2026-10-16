import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

class LabDeviceSimulator:
    def __init__(self, sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9):
        self.fs = sample_rate
        self.rs = symbol_rate
        self.fc = carrier_freq
        self.sps = int(sample_rate / symbol_rate)
        
        # 实验室设备通常有标准的成形滤波
        self.rrc_taps = self._generate_rrc(alpha=0.35, span=10, sps=self.sps)

    def _generate_rrc(self, alpha, span, sps):
        t = np.arange(-span*sps, span*sps+1) / sps
        t[t == 0] = 1e-8
        t[np.abs(t) == 1/(4*alpha)] += 1e-8
        num = np.sin(np.pi*t*(1-alpha)) + 4*alpha*t*np.cos(np.pi*t*(1+alpha))
        den = np.pi*t*(1-(4*alpha*t)**2)
        return num / den

    def generate_packet(self, mod_type, num_symbols, ppm_error, iq_gain_db, iq_phase_deg, snr_db):
        """
        [全流程仿真]: 
        基带生成 -> 硬件IQ损伤 -> 成形滤波 -> 晶振频偏 -> 高斯噪声
        """
        # 1. 调制映射 (Ideal Symbols)
        syms = self._modulate(mod_type, num_symbols)
        
        # 2. 上采样
        sig_upsampled = np.zeros(num_symbols * self.sps, dtype=complex)
        
        if mod_type == 'OQPSK':
            # OQPSK: I/Q 错开半个符号
            i_sym = syms.real
            q_sym = syms.imag
            sig_i = np.zeros_like(sig_upsampled, dtype=float); sig_i[::self.sps] = i_sym
            sig_q = np.zeros_like(sig_upsampled, dtype=float); sig_q[::self.sps] = q_sym
            sig_q = np.roll(sig_q, self.sps // 2)
            sig_base = sig_i + 1j * sig_q
        else:
            sig_upsampled[::self.sps] = syms
            sig_base = sig_upsampled

        # 3. [关键] 施加发射机 IQ 不平衡 (VSG 设备特征)
        # 即使距离短，设备内部的模拟调制器也不完美
        sig_impaired = self._apply_iq_imbalance(sig_base, iq_gain_db, iq_phase_deg)

        # 4. 成形滤波
        sig_shaped = scipy.signal.convolve(sig_impaired, self.rrc_taps, mode='same')

        # 5. [关键] 施加频偏 (即使短距离，晶振不共地就有频偏)
        # 物理公式: delta_f = fc * ppm * 1e-6
        cfo_hz = self.fc * ppm_error * 1e-6
        t = np.arange(len(sig_shaped)) / self.fs
        rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
        sig_rotated = sig_shaped * rotation

        # 6. 施加极低的热噪声 (模拟短距离高信噪比)
        # 实验室环境 SNR 通常很高 (20dB+)
        sig_power = np.mean(np.abs(sig_rotated)**2)
        noise_power = sig_power / (10**(snr_db/10.0))
        noise = (np.random.randn(len(sig_rotated)) + 1j*np.random.randn(len(sig_rotated))) * np.sqrt(noise_power/2)
        
        return sig_rotated + noise

    def _modulate(self, mod_type, n):
        if mod_type == 'QPSK' or mod_type == 'OQPSK':
            return np.exp(1j * (np.pi/4 + np.random.randint(0, 4, n) * np.pi/2))
        elif mod_type == '8PSK':
            return np.exp(1j * np.random.randint(0, 8, n) * 2*np.pi/8)
        # ... 其他调制省略 ...
        return np.random.randn(n) + 1j*np.random.randn(n)

    def _apply_iq_imbalance(self, sig, gain_db, phase_deg):
        """模拟 IQ 不平衡: I路和Q路幅度和相位不一致"""
        gain = 10**(gain_db / 20.0)
        phase_rad = phase_deg * np.pi / 180.0
        
        # 模型: I 不变, Q 增益变化 + 相位旋转
        i_new = sig.real
        q_new = gain * (sig.real * np.sin(phase_rad) + sig.imag * np.cos(phase_rad))
        return i_new + 1j * q_new

# ================= 实验：探索你的设备边界 =================

def run_lab_experiment():
    sim = LabDeviceSimulator(carrier_freq=2.4e9) # 假设 2.4G
    
    # 场景 1: 完美的实验室设备 (已校准)
    # 0 ppm 频偏, 0 dB 增益误差, 30dB SNR
    sig_perfect = sim.generate_packet('QPSK', 2048, ppm_error=0, iq_gain_db=0, iq_phase_deg=0, snr_db=30)
    
    # 场景 2: 真实的未校准设备 (你的实测数据可能属于这种)
    # 5 ppm 频偏 (很小，但足以让星座图转圈)
    # 1 dB IQ增益不平衡 (导致星座图变长方形)
    # 25dB SNR
    sig_real = sim.generate_packet('QPSK', 2048, ppm_error=0.5, iq_gain_db=1.0, iq_phase_deg=2.0, snr_db=6)
    
    # 绘图对比
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(sig_perfect.real, sig_perfect.imag, s=1)
    plt.title("Ideal Lab Setup (Locked Clock)")
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.scatter(sig_real.real, sig_real.imag, s=1)
    plt.title("Realistic Lab Setup (5ppm Drift + IQ Imbal)")
    plt.axis('equal')
    
    plt.show()

if __name__ == "__main__":
    run_lab_experiment()