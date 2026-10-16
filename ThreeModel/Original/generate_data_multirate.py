#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_data_multirate.py — 多符号率/采样率 AMR 数据集生成器

vs 原版改进:
  1. 支持配置多个符号率 (symbol_rates) 和对应采样率 (sample_rates)
  2. 每个符号率下独立计算 SPS、RRC 滤波器、FSK 频偏等参数
  3. 所有生成的信号统一重采样到固定输出长度 SIG_LEN=1024
  4. 可模拟低符号率/低采样率下的特有损伤（量化噪声等）
  5. 数据集中保存 symbol_rate 字段，供推理时对比分析

输出: ./multirate_dataset/amr_multirate.npz
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIG                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    # ---- 固定输出长度 ----
    'signal_length':        1024,   # 模型输入长度，所有信号统一裁剪/填充到此

    # ---- 多符号率配置 ----
    # 每对 (symbol_rate, sample_rate) 对应一种采集场景
    # SPS = sample_rate / symbol_rate，应为整数或接近整数
    'symbol_rates': [
        {'sym_rate_mhz': 0.8,    'samp_rate_mhz': 3.2},   # SPS=4  (高速，模型训练主场景)
        {'sym_rate_mhz': 1.6,    'samp_rate_mhz': 6.4},   # SPS=4  (高速)
        {'sym_rate_mhz': 0.0125, 'samp_rate_mhz': 0.05},  # SPS=4  (低速，但SPS相同)
        # 可继续添加其他场景，例如:
        # {'sym_rate_mhz': 0.1,  'samp_rate_mhz': 0.8},   # SPS=8
    ],

    # ---- 每个场景的样本数 ----
    'n_per_class_per_snr':  1600,    # 每类每SNR每场景的样本数（可减小加快调试）

    # ---- SNR 范围 ----
    'snr_list_db':          list(range(-2, 38, 2)),  # 20 档

    # ---- 脉冲成形 ----
    'rolloff':              0.35,
    'rrc_taps':             101,    # RRC 滤波器抽头数（固定，相对符号率的覆盖范围会变）

    # ---- 数据集划分 ----
    'train_ratio':          0.8,
    'save_dir':             './multirate_dataset',

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
    'pa_p':                 3,

    # ---- 接收端损伤 ----
    'impairments_enable':   True,
    'cfo_max_norm':         0.001,
    'phase_noise_std':      0.010,
    'iq_gain_imb_max_db':   1.0,
    'iq_phase_imb_max_deg': 5.0,
    'dc_offset_max':        0.020,
    'timing_offset_max':    0.4,

    # ---- 低速率额外损伤 ----
    # 低采样率下 ADC 有效位数降低，量化噪声增大
    'enable_quantization_noise': True,
    'adc_bits_high_rate':   12,     # 高采样率时 ADC 位数
    'adc_bits_low_rate':    8,      # 低采样率时 ADC 位数（<0.1 MHz 采样率时启用）
    'low_rate_threshold_mhz': 0.5,  # 采样率低于此值时使用低位数 ADC

    # ---- 调制类别 ----
    'classes': ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
                '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK'],

    # ---- 可复现 ----
    'seed': 42,
}
# seed train - default - 42
# seed test - default - 24

SIG_LEN = CONFIG['signal_length']
NUM_CLS = len(CONFIG['classes'])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. RRC 滤波器                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
def design_rrc(ntaps, sps, beta):
    """Root Raised Cosine 滤波器，sps 可为浮点（非整数 SPS 场景）"""
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
    h /= np.linalg.norm(h)
    return h.astype(np.float64)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. FFT 批量卷积                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def batch_fft_convolve(signals, filt):
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
# ║  4. 场景参数计算                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def compute_scene_params(scene_cfg, global_cfg):
    """
    根据 (sym_rate_mhz, samp_rate_mhz) 计算单个场景的关键参数。

    返回 dict:
        sps         : 每符号采样数（可能非整数，取整后使用）
        num_sym     : 符号数（使得上采样后长度 >= SIG_LEN）
        rrc         : 对应 SPS 的 RRC 滤波器系数
        sym_rate    : 符号率 (Hz)
        samp_rate   : 采样率 (Hz)
        adc_bits    : ADC 位数（低采样率时降低）
    """
    sym_rate  = scene_cfg['sym_rate_mhz'] * 1e6
    samp_rate = scene_cfg['samp_rate_mhz'] * 1e6
    sps_float = samp_rate / sym_rate
    sps_int   = max(1, int(round(sps_float)))   # 四舍五入取整

    # 需要多少符号才能覆盖 SIG_LEN 个样本
    num_sym = SIG_LEN // sps_int + 10   # 多生成几个符号防止截断不够

    rrc = design_rrc(global_cfg['rrc_taps'], sps_int, global_cfg['rolloff'])

    # ADC 位数
    thr = global_cfg['low_rate_threshold_mhz']
    if global_cfg['enable_quantization_noise'] and \
       scene_cfg['samp_rate_mhz'] < thr:
        adc_bits = global_cfg['adc_bits_low_rate']
    else:
        adc_bits = global_cfg['adc_bits_high_rate']

    return {
        'sps':       sps_int,
        'sps_float': sps_float,
        'num_sym':   num_sym,
        'rrc':       rrc,
        'sym_rate':  sym_rate,
        'samp_rate': samp_rate,
        'adc_bits':  adc_bits,
        'scene_label': f"{scene_cfg['sym_rate_mhz']}M/{scene_cfg['samp_rate_mhz']}M",
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. 星座定义                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _make_psk(M):
    return np.exp(1j * 2 * np.pi * np.arange(M) / M)

def _make_square_qam(M):
    k = int(np.sqrt(M))
    levels = 2.0 * np.arange(k) - (k - 1)
    gI, gQ = np.meshgrid(levels, levels)
    pts = (gI + 1j * gQ).flatten()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))

CONSTELLATIONS = {
    'BPSK':  _make_psk(2),
    'QPSK':  _make_psk(4),
    '16QAM': _make_square_qam(16),
    '64QAM': _make_square_qam(64),
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. 调制信号生成器（依赖场景参数）                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _rand_symbols(name, n_batch, n_sym):
    c = CONSTELLATIONS[name]
    return c[np.random.randint(0, len(c), (n_batch, n_sym))]

def _gen_linear(const_name, n, sp):
    """线性调制：星座 → 上采样 → RRC 脉冲成形"""
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    syms = _rand_symbols(const_name, n, num_sym)
    up_len = num_sym * sps
    up = np.zeros((n, up_len), dtype=np.complex128)
    up[:, ::sps] = syms
    out = batch_fft_convolve(up, rrc)
    return out[:, :SIG_LEN]

def gen_bpsk(n, sp):    return _gen_linear('BPSK', n, sp)
def gen_qpsk(n, sp):    return _gen_linear('QPSK', n, sp)
def gen_16qam(n, sp):   return _gen_linear('16QAM', n, sp)
def gen_64qam(n, sp):   return _gen_linear('64QAM', n, sp)

def gen_oqpsk(n, sp):
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    bi = 2.0 * np.random.randint(0, 2, (n, num_sym)) - 1.0
    bq = 2.0 * np.random.randint(0, 2, (n, num_sym)) - 1.0
    L = num_sym * sps

    up_i = np.zeros((n, L))
    up_i[:, ::sps] = bi

    up_q = np.zeros((n, L))
    q_idx = np.arange(num_sym) * sps + sps // 2
    q_idx = q_idx[q_idx < L]
    up_q[:, q_idx] = bq[:, :len(q_idx)]

    si = batch_fft_convolve(up_i, rrc).real
    sq = batch_fft_convolve(up_q, rrc).real
    return (si + 1j * sq)[:, :SIG_LEN]

def gen_pi4dqpsk(n, sp):
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    dphi = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    idx = np.random.randint(0, 4, (n, num_sym))
    phase = np.cumsum(dphi[idx], axis=1)
    syms = np.exp(1j * phase)
    up_len = num_sym * sps
    up = np.zeros((n, up_len), dtype=np.complex128)
    up[:, ::sps] = syms
    out = batch_fft_convolve(up, rrc)
    return out[:, :SIG_LEN]

def _gen_cpfsk(n, sp, M, h_mod):
    """
    连续相位频移键控 (CPFSK)
    注意：h_mod 和 SPS 共同决定每样本的频率偏移。
    对于不同 SPS，只要 h_mod 固定，归一化调制指数不变。
    """
    sps, num_sym = sp['sps'], sp['num_sym']
    syms = np.random.randint(0, M, (n, num_sym))
    levels = 2.0 * np.arange(M) - (M - 1)
    freq_sym = levels[syms]

    freq_per_sample = h_mod * freq_sym / (2.0 * sps)
    freq_up = np.repeat(freq_per_sample, sps, axis=1)[:, :SIG_LEN]
    phase = 2 * np.pi * np.cumsum(freq_up, axis=1)
    return np.exp(1j * phase)

def gen_2fsk(n, sp):
    return _gen_cpfsk(n, sp, M=2, h_mod=1.0)

def gen_4fsk(n, sp):
    return _gen_cpfsk(n, sp, M=4, h_mod=1.0/3.0)

def gen_gmsk(n, sp, BT=0.3):
    sps, num_sym = sp['sps'], sp['num_sym']
    M, h_mod = 2, 0.5
    syms = np.random.randint(0, M, (n, num_sym))
    levels = np.array([-1.0, 1.0])
    freq_sym = levels[syms]
    freq_per_sample = h_mod * freq_sym / (2.0 * sps)
    freq_rect = np.repeat(freq_per_sample, sps, axis=1)

    T = sps
    B = BT / T * sps
    t_gauss = np.arange(-2 * sps, 2 * sps + 1)
    h_gauss = (2 * np.pi * B / np.sqrt(2 * np.pi * np.log(2))) * \
              np.exp(-2 * (np.pi * B * t_gauss / sps) ** 2 / np.log(2))
    h_gauss /= h_gauss.sum()

    freq_up = batch_fft_convolve(freq_rect, h_gauss).real[:, :SIG_LEN]
    phase = 2 * np.pi * np.cumsum(freq_up, axis=1)
    return np.exp(1j * phase)

GENERATORS = {
    'BPSK':     gen_bpsk,
    'QPSK':     gen_qpsk,
    'OQPSK':    gen_oqpsk,
    '2FSK':     gen_2fsk,
    '4FSK':     gen_4fsk,
    '16QAM':    gen_16qam,
    '64QAM':    gen_64qam,
    'Pi4DQPSK': gen_pi4dqpsk,
    'GMSK':     gen_gmsk,
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7. 量化噪声（低采样率 ADC 损伤）                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
def apply_quantization(x, n_bits):
    """
    模拟 ADC 量化噪声
    将 IQ 分量量化到 n_bits 精度
    """
    if n_bits <= 0:
        return x
    # 信号归一化到 [-1, 1]
    max_val = np.max(np.abs(x))
    if max_val < 1e-12:
        return x
    x_norm = x / max_val

    levels = 2 ** n_bits
    step = 2.0 / levels

    I_q = np.round(x_norm.real / step) * step
    Q_q = np.round(x_norm.imag / step) * step

    # 截断到 [-1, 1]
    I_q = np.clip(I_q, -1.0, 1.0)
    Q_q = np.clip(Q_q, -1.0, 1.0)

    return (I_q + 1j * Q_q) * max_val


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  8. 信道模型                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
class ChannelModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def _norm(self, x):
        pwr = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(pwr + 1e-12)

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

    def _awgn(self, x, snr_db):
        sigma2 = 10.0 ** (-snr_db / 10.0)
        noise = np.sqrt(sigma2 / 2) * \
                (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
        return x + noise

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
# ║  9. RF 损伤                                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
class RFImpairments:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply_tx(self, x):
        if self.cfg['pa_enable']:
            x = self._pa_rapp(x)
        return x

    def _pa_rapp(self, x):
        p   = self.cfg['pa_p']
        ibo = self.cfg['pa_ibo_db']
        pwr = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
        A_sat = np.sqrt(pwr * 10 ** (ibo / 10))
        r = np.abs(x)
        theta = np.angle(x)
        r_out = r / (1 + (r / (A_sat + 1e-12)) ** (2 * p)) ** (1 / (2 * p))
        return r_out * np.exp(1j * theta)

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
        N, L = x.shape
        max_cfo = self.cfg['cfo_max_norm']
        if max_cfo <= 0:
            return x
        df  = np.random.uniform(-max_cfo, max_cfo, (N, 1))
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        n   = np.arange(L).reshape(1, L)
        return x * np.exp(1j * (2 * np.pi * df * n + phi))

    def _phase_noise(self, x):
        N, L = x.shape
        sigma = self.cfg['phase_noise_std']
        if sigma <= 0:
            return x
        increments = sigma * np.random.randn(N, L)
        pn = np.cumsum(increments, axis=1)
        return x * np.exp(1j * pn)

    def _iq_imbalance(self, x):
        N, L = x.shape
        g_max = self.cfg['iq_gain_imb_max_db']
        p_max = self.cfg['iq_phase_imb_max_deg']
        if g_max <= 0 and p_max <= 0:
            return x
        eps  = 10 ** (np.random.uniform(-g_max, g_max, (N, 1)) / 20) - 1
        dphi = np.random.uniform(-p_max, p_max, (N, 1)) * np.pi / 180
        I, Q = x.real, x.imag
        I_out = (1 + eps / 2) * I
        Q_out = (1 - eps / 2) * (Q * np.cos(dphi) + I * np.sin(dphi))
        return I_out + 1j * Q_out

    def _dc_offset(self, x):
        N, L = x.shape
        mx = self.cfg['dc_offset_max']
        if mx <= 0:
            return x
        dc = np.random.uniform(-mx, mx, (N, 1)) + \
             1j * np.random.uniform(-mx, mx, (N, 1))
        return x + dc

    def _timing_offset(self, x):
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
# ║  10. 主生成流水线                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def generate_all(cfg=CONFIG):
    np.random.seed(cfg['seed'])

    channel     = ChannelModel(cfg)
    impairments = RFImpairments(cfg)

    classes  = cfg['classes']
    snr_list = cfg['snr_list_db']
    n_per    = cfg['n_per_class_per_snr']
    scenes   = cfg['symbol_rates']

    # 预计算各场景参数
    scene_params = [compute_scene_params(s, cfg) for s in scenes]

    print("\n  场景参数概览:")
    for i, sp in enumerate(scene_params):
        print(f"    场景 {i}: {sp['scene_label']}  "
              f"SPS={sp['sps']}  NumSym={sp['num_sym']}  "
              f"ADC={sp['adc_bits']}bit")

    all_iq, all_y, all_snr, all_sr, all_scene = [], [], [], [], []

    total = len(scenes) * len(classes) * len(snr_list)
    pbar  = tqdm(total=total, desc='Generating', ncols=95)

    for sc_idx, (scene_cfg, sp) in enumerate(zip(scenes, scene_params)):
        for cls_idx, cls_name in enumerate(classes):
            gen_fn = GENERATORS[cls_name]

            for snr_db in snr_list:
                # ① 生成基带信号（使用当前场景 SPS/RRC）
                clean = gen_fn(n_per, sp)           # (N, SIG_LEN) complex

                # ② TX 损伤 (PA)
                clean = impairments.apply_tx(clean)

                # ③ 信道 (衰落 + AWGN)
                noisy = channel.apply(clean, snr_db)

                # ④ 量化噪声（低采样率 ADC 损伤）
                if cfg['enable_quantization_noise'] and sp['adc_bits'] < 12:
                    noisy = apply_quantization(noisy, sp['adc_bits']).astype(np.complex64)

                # ⑤ RX 损伤 (CFO, 相噪, IQ, DC, 时偏)
                noisy = impairments.apply_rx(noisy)

                # ⑥ 转 IQ: (N, 2, SIG_LEN)
                iq = np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)

                all_iq.append(iq)
                all_y.extend([cls_idx] * n_per)
                all_snr.extend([float(snr_db)] * n_per)
                all_sr.extend([sp['sym_rate']] * n_per)       # 符号率 (Hz)
                all_scene.extend([sc_idx] * n_per)

                pbar.set_postfix(
                    scene=sp['scene_label'],
                    cls=cls_name,
                    snr=f'{snr_db:+d}dB'
                )
                pbar.update(1)

    pbar.close()

    X     = np.concatenate(all_iq, axis=0)
    Y     = np.array(all_y,    dtype=np.int64)
    SNR   = np.array(all_snr,  dtype=np.float32)
    SR    = np.array(all_sr,   dtype=np.float32)    # 符号率
    SCENE = np.array(all_scene, dtype=np.int64)     # 场景索引

    print(f'\n  总样本: {len(Y):,}   X: {X.shape}')
    return X, Y, SNR, SR, SCENE


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  11. 保存                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
def split_and_save(X, Y, SNR, SR, SCENE, cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    idx = np.arange(len(Y))
    i_tr, i_te = train_test_split(
        idx, test_size=1 - cfg['train_ratio'],
        random_state=cfg['seed'], stratify=Y
    )

    # 场景 label 字符串
    scene_labels = [
        f"{s['sym_rate_mhz']}M_sym_{s['samp_rate_mhz']}M_samp"
        for s in cfg['symbol_rates']
    ]

    path = os.path.join(cfg['save_dir'], 'amr_multirate_3.npz')
    np.savez_compressed(
        path,
        X_train=X[i_tr], Y_train=Y[i_tr],
        X_test=X[i_te],  Y_test=Y[i_te],
        snr_test=SNR[i_te],
        sr_test=SR[i_te],
        scene_test=SCENE[i_te],
        snr_train=SNR[i_tr],
        sr_train=SR[i_tr],
        scene_train=SCENE[i_tr],
        classes=np.array(cfg['classes']),
        snr_list=np.array(cfg['snr_list_db'], dtype=np.float32),
        scene_labels=np.array(scene_labels),
    )

    sz = os.path.getsize(path) / 1024 / 1024
    print(f'  已保存 → {path}')
    print(f'    Train: {len(i_tr):,}  Test: {len(i_te):,}  Size: {sz:.1f} MB')
    return path


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  12. MAIN                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    n_scenes = len(CONFIG['symbol_rates'])
    n_total  = (n_scenes * NUM_CLS *
                len(CONFIG['snr_list_db']) * CONFIG['n_per_class_per_snr'])

    print('=' * 65)
    print('  Multi-Rate AMR Dataset Generator')
    print(f'  Classes:     {NUM_CLS}')
    print(f'  Scenes:      {n_scenes}  (各场景有独立 SPS/RRC/ADC)')
    print(f'  SNR range:   {CONFIG["snr_list_db"][0]}~{CONFIG["snr_list_db"][-1]} dB '
          f'({len(CONFIG["snr_list_db"])} steps)')
    print(f'  Per cls/SNR/scene: {CONFIG["n_per_class_per_snr"]}')
    print(f'  Total:       {n_total:,}')
    print(f'  Channel:     {CONFIG["channel_type"]}')
    print(f'  Quant Noise: {"ON" if CONFIG["enable_quantization_noise"] else "OFF"}')
    print('=' * 65)

    X, Y, SNR, SR, SCENE = generate_all()
    path = split_and_save(X, Y, SNR, SR, SCENE)

    print('\n✓ Done!')
