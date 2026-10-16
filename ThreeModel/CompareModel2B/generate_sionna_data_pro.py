#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_data_pro.py — Professional-Grade AMR Dataset Generator (Pure NumPy)

vs 基础版改进:
  1. 全矢量化批处理（FFT 卷积，无逐样本循环）
  2. 正确几何形状的 QAM/PSK 星座（Cross-QAM）
  3. 多径频率选择性衰落信道（指数功率时延谱）
  4. 完整 RF 损伤链:
     ● PA 非线性 (Rapp SSPA)  — 发射端
     ● 载波频偏 CFO           — 收发端振荡器失配
     ● 相位噪声 (Wiener)      — 振荡器不稳定
     ● IQ 不平衡 (增益+相位)  — 接收机混频器
     ● DC 偏移               — ADC 直流泄漏
     ● 分数采样时偏           — 采样钟不同步
  5. 正确的损伤施加顺序: TX(PA) → 信道 → RX(CFO/相噪/IQ/DC/时偏)
  6. 单一 CONFIG 字典控制所有参数
  7. 可复现 (seed)

输出: sionna_dataset/sionna_amr_data.npz
依赖: numpy, scikit-learn, tqdm, (可选) matplotlib
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION — 所有参数集中于此                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    # ---- 信号参数 ----
    'signal_length':        1024,
    'sps':                  4,
    'rolloff':              0.35,
    'rrc_taps':             101,

    # ---- 数据集规模 ----
    'n_per_class_per_snr':  100,
    'snr_list_db':          list(range(0, 30, 2)),   # 20 档 (-2, 38, 1)
    'train_ratio':          0.8,
    'save_dir':             './finetune_dataset',  # sionna_dataset

    # ---- 信道 ----
    'channel_type':         'mixed',
    'mixed_weights':        {
        'awgn':             0.30,
        'flat_rayleigh':    0.20,
        'flat_rician':      0.15,
        'multipath':        0.35,
    },
    'rician_k_db':          10.0,
    'multipath_taps':       8,
    'multipath_tau_rms':    3.0,

    # ---- 发射端损伤 ----
    'pa_enable':            True,
    'pa_ibo_db':            8.0,
    'pa_p':                 3,          # Rapp 模型平滑因子

    # ---- 接收端损伤 ----
    'impairments_enable':   True,
    'cfo_max_norm':         0.002,      # 最大归一化频偏 Δf/fs
    'phase_noise_std':      0.015,       # Wiener 过程 σ (rad/sample)
    'iq_gain_imb_max_db':   1.5,        # 最大 IQ 增益不平衡 (dB)
    'iq_phase_imb_max_deg': 5.0,        # 最大 IQ 相位不平衡 (°)
    'dc_offset_max':        0.020,       # 最大 DC 偏移 (相对值)
    'timing_offset_max':    0.4,        # 最大分数采样偏移 (样点)

    # ---- 新增：模拟硬件 ADC 采集时的增益波动 ----
    'adc_bit_depth':    16,
    'gain_range':   [0.003, 0.015],

    # ---- 调制类别 ----
    'classes': ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM',
                '128QAM', '256QAM', 'MSK'],

    # ---- 可复现性 ----
    'seed':                 42,
}

SIG_LEN    = CONFIG['signal_length']
SPS        = CONFIG['sps']
NUM_SYM    = SIG_LEN // SPS
NUM_CLS    = len(CONFIG['classes'])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. RRC 脉冲成形滤波器                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
def design_rrc(ntaps, sps, beta):
    """Root Raised Cosine 滤波器（教科书标准公式）"""
    T = float(sps)
    n = np.arange(-(ntaps // 2), ntaps // 2 + 1, dtype=np.float64)
    h = np.zeros_like(n)
    for i, t in enumerate(n):
        if abs(t) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(t) - T / (4.0 * beta)) < 1e-12:
            h[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * t / T * (1 - beta)) +
                   4 * beta * t / T * np.cos(np.pi * t / T * (1 + beta)))
            den = np.pi * t / T * (1 - (4 * beta * t / T) ** 2)
            h[i] = num / den
    return (h / np.linalg.norm(h)).astype(np.float64)


_RRC = design_rrc(CONFIG['rrc_taps'], SPS, CONFIG['rolloff'])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. FFT 批量卷积（替代逐样本 np.convolve）                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
def batch_fft_convolve(signals, filt):
    """
    矢量化 FFT 卷积：一次处理整个 batch

    Args:
        signals: (N, L) complex/real
        filt:    (K,)   滤波器系数
    Returns:
        (N, L)  mode='same'
    """
    N, L = signals.shape
    K = len(filt)
    fft_len = 1
    while fft_len < L + K - 1:
        fft_len <<= 1
    X = np.fft.fft(signals, n=fft_len, axis=1)
    H = np.fft.fft(filt, n=fft_len)
    Y = np.fft.ifft(X * H[np.newaxis, :], axis=1)
    start = (K - 1) // 2
    return Y[:, start:start + L]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. 星座定义                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _make_psk(M):
    return np.exp(1j * 2 * np.pi * np.arange(M) / M)


def _make_square_qam(M):
    """方形 QAM: 4, 16, 64, 256"""
    k = int(np.sqrt(M))
    assert k * k == M
    levels = 2.0 * np.arange(k) - (k - 1)
    gI, gQ = np.meshgrid(levels, levels)
    pts = (gI + 1j * gQ).flatten()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _make_cross_qam(M):
    """
    Cross-QAM: 32, 128

    32-QAM:  6×6 格点去掉 4 角 = 32
    128-QAM: 12×12 格点取功率最低 128 个
    """
    if M == 32:
        v = np.array([-5., -3., -1., 1., 3., 5.])
        gI, gQ = np.meshgrid(v, v)
        pts = (gI + 1j * gQ).flatten()
        mask = ~((np.abs(gI.flatten()) == 5) & (np.abs(gQ.flatten()) == 5))
        pts = pts[mask]
    elif M == 128:
        v = 2.0 * np.arange(12) - 11
        gI, gQ = np.meshgrid(v, v)
        pts = (gI + 1j * gQ).flatten()
        idx = np.argsort(np.abs(pts) ** 2)[:128]
        pts = pts[idx]
    else:
        raise ValueError(f"Cross-QAM 未定义 M={M}")
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


# 预构建所有星座（程序启动时执行一次）
CONSTELLATIONS = {
    'BPSK':   _make_psk(2),
    'QPSK':   _make_psk(4),
    '8PSK':   _make_psk(8),
    '16QAM':  _make_square_qam(16),
    '32QAM':  _make_cross_qam(32),
    '64QAM':  _make_square_qam(64),
    '128QAM': _make_cross_qam(128),
    '256QAM': _make_square_qam(256),
}


def _rand_symbols(name, n_batch, n_sym=NUM_SYM):
    """从指定星座随机选符号 → (n_batch, n_sym) complex"""
    c = CONSTELLATIONS[name]
    return c[np.random.randint(0, len(c), (n_batch, n_sym))]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. 信道模型                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
class ChannelModel:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def _norm(self, x):
        pwr = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(pwr + 1e-12)

    # ---------- 衰落类型 ----------
    def _flat_rayleigh(self, x):
        N = x.shape[0]
        h = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)
        return x * h

    def _flat_rician(self, x):
        N = x.shape[0]
        K = 10 ** (self.cfg['rician_k_db'] / 10)
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        h_los  = np.sqrt(K / (K + 1)) * np.exp(1j * phi)
        h_nlos = np.sqrt(1 / (2 * (K + 1))) * \
                 (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
        return x * (h_los + h_nlos)

    def _multipath(self, x):
        """
        多径频率选择性衰落
        指数功率时延谱: PDP(k) = exp(-k / τ_rms)
        每条径独立 Rayleigh 衰落
        """
        N, L = x.shape
        n_taps  = self.cfg['multipath_taps']
        tau_rms = self.cfg['multipath_tau_rms']

        pdp = np.exp(-np.arange(n_taps, dtype=np.float64) / tau_rms)
        pdp /= pdp.sum()

        h = np.zeros((N, n_taps), dtype=np.complex128)
        for k in range(n_taps):
            h[:, k] = np.sqrt(pdp[k] / 2) * \
                      (np.random.randn(N) + 1j * np.random.randn(N))

        y = np.zeros_like(x)
        for k in range(n_taps):
            if k == 0:
                y += h[:, 0:1] * x
            else:
                y[:, k:] += h[:, k:k + 1] * x[:, :-k]
        return y

    # ---------- 噪声 ----------
    def _awgn(self, x, snr_db):
        sigma2 = 10.0 ** (-snr_db / 10.0)
        noise = np.sqrt(sigma2 / 2) * \
                (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
        return x + noise

    # ---------- 主接口 ----------
    def apply(self, signals, snr_db):
        x = self._norm(signals)
        ch = self.cfg['channel_type']

        if ch == 'mixed':
            x = self._apply_mixed(x)
        elif ch == 'flat_rayleigh':
            x = self._flat_rayleigh(x)
        elif ch == 'flat_rician':
            x = self._flat_rician(x)
        elif ch == 'multipath':
            x = self._multipath(x)

        return self._awgn(x, snr_db).astype(np.complex64)

    def _apply_mixed(self, x):
        N = x.shape[0]
        w = self.cfg['mixed_weights']
        types, probs = list(w.keys()), list(w.values())
        assign = np.random.choice(types, size=N, p=probs)

        y = x.copy()
        for ch in types:
            m = (assign == ch)
            if not m.any():
                continue
            sub = x[m]
            if ch == 'flat_rayleigh':
                sub = self._flat_rayleigh(sub)
            elif ch == 'flat_rician':
                sub = self._flat_rician(sub)
            elif ch == 'multipath':
                sub = self._multipath(sub)
            y[m] = sub
        return y


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. RF 损伤                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
class RFImpairments:
    """
    施加顺序（符合实际硬件链路）:
      TX: PA 非线性
      RX: CFO → 相位噪声 → IQ 不平衡 → DC 偏移 → 采样时偏
    """
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    # ======================== TX 侧 ========================
    def apply_tx(self, x):
        if self.cfg['pa_enable']:
            x = self._pa_rapp(x)
        return x

    def _pa_rapp(self, x):
        """
        Rapp SSPA 模型 (仅 AM/AM，无 AM/PM)
        G(r) = r / [1 + (r/A_sat)^{2p}]^{1/(2p)}
        """
        p = self.cfg['pa_p']
        ibo = self.cfg['pa_ibo_db']
        pwr = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
        A_sat = np.sqrt(pwr * 10 ** (ibo / 10))

        r = np.abs(x)
        theta = np.angle(x)
        r_out = r / (1 + (r / (A_sat + 1e-12)) ** (2 * p)) ** (1 / (2 * p))
        return r_out * np.exp(1j * theta)

    # ======================== RX 侧 ========================
    def apply_rx(self, x):
        if not self.cfg['impairments_enable']:
            return x
        x = self._cfo(x)
        x = self._phase_noise(x)
        x = self._iq_imbalance(x)
        x = self._dc_offset(x)
        x = self._timing_offset(x)
        return x

    def _cfo(self, x):
        """载波频偏: x · exp(j·(2π·Δf·n + φ₀))"""
        N, L = x.shape
        max_cfo = self.cfg['cfo_max_norm']
        if max_cfo <= 0:
            return x
        df  = np.random.uniform(-max_cfo, max_cfo, (N, 1))
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        n   = np.arange(L).reshape(1, L)
        return x * np.exp(1j * (2 * np.pi * df * n + phi))

    def _phase_noise(self, x):
        """
        振荡器相位噪声 (Wiener 过程)
        φ[n] = φ[n-1] + N(0, σ²)
        """
        N, L = x.shape
        sigma = self.cfg['phase_noise_std']
        if sigma <= 0:
            return x
        increments = sigma * np.random.randn(N, L)
        pn = np.cumsum(increments, axis=1)
        return x * np.exp(1j * pn)

    def _iq_imbalance(self, x):
        """
        IQ 不平衡 (直接变频接收机模型)

        I_out = (1 + ε/2) · I
        Q_out = (1 − ε/2) · [Q·cos(Δφ) + I·sin(Δφ)]

        ε: 增益不平衡 (线性)
        Δφ: 相位不平衡 (rad)
        """
        N, L = x.shape
        g_max = self.cfg['iq_gain_imb_max_db']
        p_max = self.cfg['iq_phase_imb_max_deg']
        if g_max <= 0 and p_max <= 0:
            return x

        eps  = 10 ** (np.random.uniform(-g_max, g_max, (N, 1)) / 20) - 1  # dB → 线性偏差
        dphi = np.random.uniform(-p_max, p_max, (N, 1)) * np.pi / 180

        I, Q = x.real, x.imag
        I_out = (1 + eps / 2) * I
        Q_out = (1 - eps / 2) * (Q * np.cos(dphi) + I * np.sin(dphi))
        return I_out + 1j * Q_out

    def _dc_offset(self, x):
        """ADC 直流泄漏"""
        N, L = x.shape
        mx = self.cfg['dc_offset_max']
        if mx <= 0:
            return x
        dc = np.random.uniform(-mx, mx, (N, 1)) + \
             1j * np.random.uniform(-mx, mx, (N, 1))
        return x + dc

    def _timing_offset(self, x):
        """分数采样时偏 (线性插值)"""
        N, L = x.shape
        mx = self.cfg['timing_offset_max']
        if mx <= 0:
            return x
        offsets = np.random.uniform(-mx, mx, N)
        t = np.arange(L, dtype=np.float64)
        y = np.empty_like(x)
        for i in range(N):
            t_new = np.clip(t + offsets[i], 0, L - 1)
            y[i] = np.interp(t_new, t, x[i].real) + \
                   1j * np.interp(t_new, t, x[i].imag)
        return y


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7. 调制信号生成器（全部矢量化）                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

# -------- A. 通用线性调制: 星座 → 上采样 → RRC --------
def _gen_linear(const_name, n):
    syms = _rand_symbols(const_name, n)        # (N, NUM_SYM)
    up = np.zeros((n, NUM_SYM * SPS), dtype=np.complex128)
    up[:, ::SPS] = syms
    return batch_fft_convolve(up, _RRC)[:, :SIG_LEN]


def gen_bpsk(n):    return _gen_linear('BPSK', n)
def gen_qpsk(n):    return _gen_linear('QPSK', n)
def gen_8psk(n):    return _gen_linear('8PSK', n)
def gen_16qam(n):   return _gen_linear('16QAM', n)
def gen_32qam(n):   return _gen_linear('32QAM', n)
def gen_64qam(n):   return _gen_linear('64QAM', n)
def gen_128qam(n):  return _gen_linear('128QAM', n)
def gen_256qam(n):  return _gen_linear('256QAM', n)


# -------- B. OQPSK: Q 路半符号延迟 --------
def gen_oqpsk(n):
    bi = 2.0 * np.random.randint(0, 2, (n, NUM_SYM)) - 1.0
    bq = 2.0 * np.random.randint(0, 2, (n, NUM_SYM)) - 1.0
    L = NUM_SYM * SPS

    up_i = np.zeros((n, L))
    up_i[:, ::SPS] = bi

    up_q = np.zeros((n, L))
    q_idx = np.arange(NUM_SYM) * SPS + SPS // 2
    q_idx = q_idx[q_idx < L]
    up_q[:, q_idx] = bq[:, :len(q_idx)]

    si = batch_fft_convolve(up_i, _RRC).real
    sq = batch_fft_convolve(up_q, _RRC).real
    return (si + 1j * sq)[:, :SIG_LEN]


# -------- C. π/4-DQPSK: 差分编码 + π/4 旋转 --------
def gen_pi4dqpsk(n):
    dphi = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    idx = np.random.randint(0, 4, (n, NUM_SYM))
    phase = np.cumsum(dphi[idx], axis=1)        # 矢量化差分累积
    syms = np.exp(1j * phase)

    up = np.zeros((n, NUM_SYM * SPS), dtype=np.complex128)
    up[:, ::SPS] = syms
    return batch_fft_convolve(up, _RRC)[:, :SIG_LEN]


# -------- D. 2ASK (OOK) --------
def gen_2ask(n):
    bits = np.random.randint(0, 2, (n, NUM_SYM)).astype(np.float64)
    up = np.zeros((n, NUM_SYM * SPS), dtype=np.complex128)
    up[:, ::SPS] = bits
    return batch_fft_convolve(up, _RRC)[:, :SIG_LEN]


# -------- E. CPFSK 族 (矢量化，无逐符号循环) --------
def _gen_cpfsk(n, M, h_mod):
    """
    通用连续相位频移键控 (矢量化)

    M:     调制阶数 (2, 4)
    h_mod: 调制指数
    """
    syms = np.random.randint(0, M, (n, NUM_SYM))
    levels = 2.0 * np.arange(M) - (M - 1)       # [-1,1] / [-3,-1,1,3]
    freq_sym = levels[syms]                       # (N, NUM_SYM)

    # 归一化频偏: phase_per_symbol = π · h · a_k
    # phase_per_sample = π · h · a_k / SPS
    freq_per_sample = h_mod * freq_sym / (2.0 * SPS)   # 归一化到 cycles/sample

    # 上采样: repeat
    freq_up = np.repeat(freq_per_sample, SPS, axis=1)[:, :SIG_LEN]

    # 累积相位
    phase = 2 * np.pi * np.cumsum(freq_up, axis=1)
    return np.exp(1j * phase)

def gen_4fsk(n):
    """4FSK: 使用与基础版相同的调制指数 h=1/3"""
    M = 4
    h_mod = 1.0 / 3.0  # ← 关键修复

    syms = np.random.randint(0, M, (n, NUM_SYM))
    levels = 2.0 * np.arange(M) - (M - 1)  # [-3,-1,1,3]
    freq_sym = levels[syms]

    freq_per_sample = h_mod * freq_sym / (2.0 * SPS)
    freq_up = np.repeat(freq_per_sample, SPS, axis=1)[:, :SIG_LEN]
    phase = 2 * np.pi * np.cumsum(freq_up, axis=1)
    return np.exp(1j * phase)

def gen_2fsk(n):  return _gen_cpfsk(n, M=2, h_mod=1.0)
def gen_msk(n):   return _gen_cpfsk(n, M=2, h_mod=0.5)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  8. 生成器注册表                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
GENERATORS = {
    'BPSK': gen_bpsk,    'QPSK': gen_qpsk,      '8PSK': gen_8psk,
    'OQPSK': gen_oqpsk,  '2FSK': gen_2fsk,      '4FSK': gen_4fsk,
    '16QAM': gen_16qam,  '2ASK': gen_2ask,      'Pi4DQPSK': gen_pi4dqpsk,
    '64QAM': gen_64qam,  '32QAM': gen_32qam,    '128QAM': gen_128qam,
    '256QAM': gen_256qam, 'MSK': gen_msk,
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  9. 主生成流水线                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def generate_all(cfg=CONFIG):
    np.random.seed(cfg['seed'])

    channel = ChannelModel(cfg)
    impairments = RFImpairments(cfg)

    classes  = cfg['classes']
    snr_list = cfg['snr_list_db']
    n_per    = cfg['n_per_class_per_snr']

    all_iq, all_y, all_snr = [], [], []
    total = len(classes) * len(snr_list)
    pbar = tqdm(total=total, desc='Generating', ncols=90)

    for cls_idx, cls_name in enumerate(classes):
        gen_fn = GENERATORS[cls_name]

        for snr_db in snr_list:
            # ① 生成干净基带信号
            clean = gen_fn(n_per)                        # (N, SIG_LEN) complex

            # ② TX 损伤 (PA)
            clean = impairments.apply_tx(clean)

            # ③ 信道 (衰落 + AWGN)
            noisy = channel.apply(clean, snr_db)

            # # ④ RX 损伤 (CFO, 相噪, IQ不平衡, DC, 时偏)
            # noisy = impairments.apply_rx(noisy)

            # # ⑤ 转 IQ: (N, 2, L)
            # iq = np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)

            # ④ RX 损伤 (CFO, 相噪, IQ不平衡, DC, 时偏)
            noisy = impairments.apply_rx(noisy)

            # ──── 新增：模拟不同硬件增益下的 16-bit ADC 量化噪声 ────
            # 1. 随机赋予一个类似于真实信号（A或B）的幅度增益
            #gain = np.random.uniform(CONFIG['gain_range'][0], CONFIG['gain_range'][1], size=(n_per, 1))
            log_low = np.log10(CONFIG['gain_range'][0])
            log_high = np.log10(CONFIG['gain_range'][1])
            gain = 10 ** np.random.uniform(log_low, log_high, size=(n_per, 1))
            noisy_scaled = noisy * gain

            # 2. 模拟 16-bit 定点数整型量化 (值域 -32768 到 32767)
            max_val = 2 ** (CONFIG['adc_bit_depth'] - 1) - 1
            noisy_quantized = np.clip(noisy_scaled * max_val, -max_val - 1, max_val)
            # 四舍五入取整
            noisy_quantized = np.round(noisy_quantized)
            # 重新转回 float32，此时已带有了微小的量化台阶噪声
            noisy = noisy_quantized / max_val
            # ────────────────────────────────────────────────────────

            # ⑤ 转 IQ: (N, 2, L)
            iq = np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)

            all_iq.append(iq)
            all_y.extend([cls_idx] * n_per)
            all_snr.extend([float(snr_db)] * n_per)

            pbar.set_postfix(cls=cls_name, snr=f'{snr_db:+d}dB')
            pbar.update(1)

    pbar.close()

    X   = np.concatenate(all_iq, axis=0)
    Y   = np.array(all_y, dtype=np.int64)
    SNR = np.array(all_snr, dtype=np.float32)

    print(f'\n  总样本: {len(Y):,}   X: {X.shape}   '
          f'SNR: {SNR.min():.0f}~{SNR.max():.0f} dB')
    return X, Y, SNR


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  10. 保存                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
def split_and_save(X, Y, SNR, cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    idx = np.arange(len(Y))
    i_tr, i_te = train_test_split(idx, test_size=1 - cfg['train_ratio'],
                                  random_state=cfg['seed'], stratify=Y)

    path = os.path.join(cfg['save_dir'], 'SionnaA_finetune.npz')
    np.savez_compressed(path,
                        X_train=X[i_tr], Y_train=Y[i_tr],
                        X_test=X[i_te],  Y_test=Y[i_te],
                        snr_test=SNR[i_te],
                        classes=np.array(cfg['classes']),
                        snr_list=np.array(cfg['snr_list_db'], dtype=np.float32))

    sz = os.path.getsize(path) / 1024 / 1024
    print(f'  已保存 → {path}')
    print(f'    Train: {len(i_tr):,}  Test: {len(i_te):,}  Size: {sz:.1f} MB')
    return path


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  11. 验证与可视化                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
def validate(npz_path, cfg=CONFIG):
    data = np.load(npz_path, allow_pickle=True)
    cls_names = list(data['classes'])

    print('\n===== 数据统计 =====')
    for split in ['train', 'test']:
        X, Y = data[f'X_{split}'], data[f'Y_{split}']
        print(f'{split}: {X.shape}  {Y.shape}')
        for c in range(len(cls_names)):
            print(f'    [{c:2d}] {cls_names[c]:>10s}: {np.sum(Y == c):5d}')

    # ---- 信号质量检查 ----
    X_test, Y_test = data['X_test'], data['Y_test']
    print('\n===== 信号质量检查 =====')
    for c in range(len(cls_names)):
        mask = Y_test == c
        samples = X_test[mask]
        pwr = np.mean(samples[:, 0] ** 2 + samples[:, 1] ** 2, axis=-1)
        print(f'  {cls_names[c]:>10s}: '
              f'power={np.mean(pwr):.4f}±{np.std(pwr):.4f}  '
              f'I_range=[{samples[:,0].min():.3f}, {samples[:,0].max():.3f}]  '
              f'Q_range=[{samples[:,1].min():.3f}, {samples[:,1].max():.3f}]')

    # ---- 可视化 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 7, figsize=(24, 7))
        snr_test = data['snr_test']

        for c in range(len(cls_names)):
            ax = axes[c // 7, c % 7]
            mask = (Y_test == c) & (snr_test >= 10)   # 高 SNR 样本看星座
            if mask.sum() == 0:
                mask = Y_test == c
            s = X_test[mask][0]
            ax.plot(s[0, :300], s[1, :300], '.', ms=1.2, alpha=0.5)
            ax.set_title(cls_names[c], fontsize=10, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

        plt.suptitle('IQ Scatter — High SNR Samples (first 300 pts)', fontsize=13)
        plt.tight_layout()
        fig_path = os.path.join(cfg['save_dir'], 'verify_iq_scatter.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f'\n  验证图 → {fig_path}')

        # ---- 功率 vs SNR ----
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        snr_vals = sorted(set(snr_test))
        for c in range(len(cls_names)):
            mean_pwr = []
            for s in snr_vals:
                m = (Y_test == c) & (snr_test == s)
                if m.sum() > 0:
                    p = np.mean(X_test[m, 0] ** 2 + X_test[m, 1] ** 2, axis=-1)
                    mean_pwr.append(np.mean(p))
                else:
                    mean_pwr.append(np.nan)
            ax2.plot(snr_vals, mean_pwr, '.-', label=cls_names[c], ms=3)
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Mean Power')
        ax2.set_title('Signal Power vs SNR by Class')
        ax2.legend(fontsize=7, ncol=4)
        ax2.grid(True, alpha=0.3)
        fig2_path = os.path.join(cfg['save_dir'], 'verify_power_vs_snr.png')
        plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f'  功率图 → {fig2_path}')

    except ImportError:
        print('  (matplotlib 未安装，跳过画图)')


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  12. MAIN                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    n_total = NUM_CLS * len(CONFIG['snr_list_db']) * CONFIG['n_per_class_per_snr']
    print('=' * 65)
    print('  Professional-Grade AMR Dataset Generator')
    print(f'  Classes:     {NUM_CLS}')
    print(f'  SNR range:   {CONFIG["snr_list_db"][0]}~{CONFIG["snr_list_db"][-1]} dB '
          f'({len(CONFIG["snr_list_db"])} steps)')
    print(f'  Per cls/SNR: {CONFIG["n_per_class_per_snr"]}')
    print(f'  Total:       {n_total:,}')
    print(f'  SPS={SPS}  Length={SIG_LEN}  Symbols={NUM_SYM}')
    print(f'  Channel:     {CONFIG["channel_type"]}')
    print(f'  PA:          {"ON" if CONFIG["pa_enable"] else "OFF"}')
    print(f'  Impairments: {"ON" if CONFIG["impairments_enable"] else "OFF"}')
    print('=' * 65)

    X, Y, SNR = generate_all()
    path = split_and_save(X, Y, SNR)
    validate(path)

    print('\n✓ Done!')