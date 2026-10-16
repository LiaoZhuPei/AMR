import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

class UAVLinkSimulator:
    def __init__(self, sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9):
        """
        初始化无人机链路参数
        :param sample_rate: 接收机采样率 (Fs), 默认 100k
        :param symbol_rate: 符号率 (Rs), 默认 25k (SPS=4)
        :param carrier_freq: 载波频率, 默认 2.4GHz (影响频偏的绝对Hz数)
        """
        self.fs = sample_rate
        self.rs = symbol_rate
        self.fc = carrier_freq
        self.sps = int(sample_rate / symbol_rate)
        
        # RRC 滤波器 (固定的物理层标准)
        self.rrc_taps = self._generate_rrc(alpha=0.35, span=10, sps=self.sps)

    def _generate_rrc(self, alpha, span, sps):
        """生成根升余弦滤波器系数"""
        t = np.arange(-span*sps, span*sps+1) / sps
        # 避免分母为0
        t[t == 0] = 1e-8
        t[np.abs(t) == 1/(4*alpha)] += 1e-8
        
        num = np.sin(np.pi*t*(1-alpha)) + 4*alpha*t*np.cos(np.pi*t*(1+alpha))
        den = np.pi*t*(1-(4*alpha*t)**2)
        return num / den

    def transmitter(self, mod_type, num_symbols, tx_power_dbm=20):
        """
        [发射机]: 调制 -> 上采样 -> 成形滤波 -> 功率放大(非线性)
        """
        # 1. 生成符号
        if mod_type == 'QPSK':
            idx = np.random.randint(0, 4, num_symbols)
            syms = np.exp(1j * (np.pi/4 + idx * np.pi/2))
        elif mod_type == 'OQPSK':
            # OQPSK 特殊处理：I/Q 错开
            idx = np.random.randint(0, 4, num_symbols)
            base = np.exp(1j * (np.pi/4 + idx * np.pi/2))
            i = base.real
            q = base.imag
            # 上采样
            sig_i = np.zeros(num_symbols * self.sps); sig_i[::self.sps] = i
            sig_q = np.zeros(num_symbols * self.sps); sig_q[::self.sps] = q
            # Q 路延迟半个符号
            sig_q = np.roll(sig_q, self.sps // 2)
            sig_upsampled = sig_i + 1j * sig_q
        else:
            # 默认为随机复数
            syms = (np.random.randn(num_symbols) + 1j*np.random.randn(num_symbols))
            sig_upsampled = np.zeros(num_symbols * self.sps, dtype=complex)
            sig_upsampled[::self.sps] = syms

        # 对于非 OQPSK 的上采样
        if mod_type != 'OQPSK':
            sig_upsampled = np.zeros(num_symbols * self.sps, dtype=complex)
            sig_upsampled[::self.sps] = syms

        # 2. 成形滤波 (模拟带宽限制)
        sig_shaped = scipy.signal.convolve(sig_upsampled, self.rrc_taps, mode='same')

        # 3. 功率放大器 (PA) 非线性
        # 无人机为了省电和距离，PA通常工作在接近饱和区
        # 使用 Rapp 模型模拟幅度压缩
        sig_amp = np.abs(sig_shaped)
        # 归一化到饱和电平，假设输入幅度1.0对应饱和点
        sig_out = sig_shaped / (1 + sig_amp**2)**0.5
        
        return sig_out

    def channel_and_receiver(self, sig, ppm_error, snr_db):
        """
        [信道与接收机]: 晶振误差 -> 多径 -> 热噪声
        :param ppm_error: 晶振误差 (Parts Per Million), e.g., 20
        """
        n = len(sig)
        t = np.arange(n) / self.fs

        # --- 1. 频率偏差 (CFO) ---
        # 物理公式: delta_f = fc * ppm * 1e-6
        # 例如 2.4G * 20ppm = 48kHz
        cfo_hz = self.fc * ppm_error * 1e-6
        
        # 施加旋转
        rotation = np.exp(1j * 2 * np.pi * cfo_hz * t)
        sig = sig * rotation

        # --- 2. 采样率偏差 (SRO) ---
        # 同样的晶振误差会导致采样率也偏 ppm
        # 这会导致波形长度伸缩，模拟真实的"非整数采样"
        actual_fs = self.fs * (1 + ppm_error * 1e-6)
        ratio = self.fs / actual_fs # 重采样比例
        
        # 使用多相滤波重采样 (模拟 ADC 采样过程)
        new_len = int(n * ratio)
        sig = scipy.signal.resample(sig, new_len)
        
        # 截断或补齐回原长度 (接收机 buffer 大小固定)
        if len(sig) > n: sig = sig[:n]
        else: sig = np.pad(sig, (0, n - len(sig)), 'wrap')

        # --- 3. 莱斯信道 (低空视距 + 地面反射) ---
        # 主径强，反射径弱
        path_delays = [0, 2] # 采样点延迟
        path_gains = [1.0, 0.2 * np.exp(1j * np.random.uniform(0, 2*np.pi))] 
        
        sig_multipath = np.zeros_like(sig)
        for d, g in zip(path_delays, path_gains):
            sig_multipath += g * np.roll(sig, d)
            
        # --- 4. 接收机热噪声 ---
        sig_power = np.mean(np.abs(sig_multipath)**2)
        noise_power = sig_power / (10**(snr_db/10.0))
        noise = (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig))) * np.sqrt(noise_power/2)
        
        return sig_multipath + noise, cfo_hz

# ==========================================
# 3. 控制变量法：探索频偏边界 (Boundary Exploration)
# ==========================================

def run_experiment():
    # 实例化仿真器：2.4GHz 无人机链路
    sim = UAVLinkSimulator(sample_rate=100000, symbol_rate=25000, carrier_freq=2.4e9)
    
    # 设定实验范围：晶振误差从 0 PPM 到 50 PPM (消费级晶振极限)
    ppm_values = [0, 5, 10, 20, 30, 40, 50]
    
    print(f"{'PPM':<10} | {'CFO (Hz)':<15} | {'Relative CFO (Normalized)'}")
    print("-" * 50)
    
    for ppm in ppm_values:
        # 生成一个样本
        tx_sig = sim.transmitter('QPSK', num_symbols=256)
        rx_sig, real_cfo = sim.channel_and_receiver(tx_sig, ppm_error=ppm, snr_db=15)
        
        # 计算归一化频偏 = CFO / Fs
        # 这是一个关键指标，通常神经网络能处理 < 0.05，传统算法能处理 < 0.1
        norm_cfo = real_cfo / sim.fs
        
        print(f"{ppm:<10} | {real_cfo:<15.1f} | {norm_cfo:.5f}")
        
        # [这里可以插入你的模型推理代码]
        # pred = model(rx_sig)
        # 观察在哪个 PPM 值开始，模型识别率下降

    print("-" * 50)
    print("实验结论参考：")
    print("1. 20 PPM (典型值) -> 48 kHz -> 归一化 0.48")
    print("   注意！归一化频偏 0.48 极其巨大！接近 0.5 (奈奎斯特极限)。")
    print("   如果不做校正，QPSK 星座图每两个采样点就转了 180 度，完全混叠。")
    print("   这说明：对于 2.4GHz 的窄带(100k)信号，即使是普通的 20ppm 晶振，")
    print("   其产生的绝对频偏相对于采样率来说也是毁灭性的。")

if __name__ == "__main__":
    run_experiment()