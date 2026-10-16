#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_data_multirate_dynamic.py — 动态物理参数版多符号率 AMR 数据集生成器

相对上一版（physical）的核心改进：
  所有信道/损伤参数不再是固定的物理值，而是在每个 batch 前
  从物理合理范围的分布中独立采样，保持物理单位不变，换算逻辑不变。

随机化策略（见 PhysicalParamSampler）：
  ┌─────────────────────┬──────────────────────────────┬─────────────────┐
  │ 参数                │ 采样范围（物理单位）          │ 分布            │
  ├─────────────────────┼──────────────────────────────┼─────────────────┤
  │ 多径时延 τ_rms      │ [0.05, 2.0] μs               │ log-uniform     │
  │ 多径抽头数          │ [2, 12] 整数                 │ log-uniform     │
  │ 载波频偏 CFO        │ 从4档精度等级随机选           │ 分类+uniform    │
  │ 相位噪声 PSD        │ [1e-9, 1e-5] rad²/Hz         │ log-uniform     │
  │ 时间偏移上限        │ [0.05, 0.5] T_sym            │ uniform → sub   │
  │ IQ 幅度不平衡上限   │ [0.1, 3.0] dB                │ uniform → sub   │
  │ IQ 相位不平衡上限   │ [0.5, 10.0] deg              │ uniform → sub   │
  │ DC 偏置上限         │ [0.002, 0.05]                │ log-uniform     │
  │ PA IBO              │ [3, 15] dB                   │ uniform         │
  │ PA Rapp 阶数 p      │ {1, 2, 3, 4}                 │ 离散均匀        │
  │ 莱斯 K 因子         │ [0, 20] dB                   │ uniform         │
  │ 运动速度 v          │ [0, 120] km/h                │ log-uniform+0   │
  │ 载波频率 fc         │ [400, 6000] MHz              │ log-uniform     │
  │ 信道混合权重        │ Dirichlet(α=[2,1,1,3])       │ Dirichlet       │
  └─────────────────────┴──────────────────────────────┴─────────────────┘

  "log-uniform"  = 在对数域均匀，等价于指数/数量级上均匀覆盖
  "两级随机"     = 先采样上限 M，再在 [0, M] 均匀采样实例值
  "Dirichlet"    = 归一化正随机向量，模拟不同信道条件占比的变化

换算逻辑（与上一版相同，在 compute_scene_params 中完成）：
  f_s = samp_rate (Hz)
  τ_rms_samp  = τ_rms_us  × 1e-6 × f_s
  cfo_norm    = cfo_hz    / f_s
  σ_pn/sample = sqrt(psd  × f_s)
  t_off_samp  = ε_sym     × SPS
  f_d_norm    = (v/c×fc)  / f_s
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIG — 只保留结构性参数，不再出现损伤具体数值                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    'signal_length':       1024,

    'symbol_rates': [
        {'sym_rate_mhz': 0.8,    'samp_rate_mhz': 3.2},
        {'sym_rate_mhz': 1.6,    'samp_rate_mhz': 6.4},
        {'sym_rate_mhz': 0.0125, 'samp_rate_mhz': 0.05},
    ],

    'n_per_class_per_snr': 1600,
    'snr_list_db':         list(range(-2, 38, 2)),
    'rolloff':             0.35,
    'rrc_taps':            101,
    'train_ratio':         0.8,
    'save_dir':            './multirate_dataset',

    # ── 信道结构性开关 ──
    'channel_type':        'mixed',
    'pa_enable':           True,
    'impairments_enable':  True,
    'doppler_enable':      True,

    # ── ADC 量化噪声 ──
    'enable_quantization_noise': True,
    'adc_bits_high_rate':  12,
    'adc_bits_low_rate':   8,
    'low_rate_threshold_mhz': 0.5,

    'classes': ['BPSK', 'QPSK', 'OQPSK', '2FSK', '4FSK',
                '16QAM', '64QAM', 'Pi4DQPSK', 'GMSK'],
    'seed': 42,
}

SIG_LEN = CONFIG['signal_length']
NUM_CLS = len(CONFIG['classes'])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. PhysicalParamSampler — 核心：每个 batch 独立采样物理参数         ║
# ╚══════════════════════════════════════════════════════════════════════╝
class PhysicalParamSampler:
    """
    所有损伤/信道参数以物理单位从合理范围内随机采样。
    每次调用 sample() 返回一个参数字典，供信道和损伤模型使用。
    换算（物理单位 → 采样域）在 compute_scene_params 中完成。
    """

    # CFO 精度档位（Hz）：对应不同晶振/补偿质量
    CFO_TIERS_HZ = [100.0, 500.0, 2000.0, 8000.0]

    # Dirichlet 浓度参数，控制信道类型权重的集中程度
    # α 越大 → 对应类型权重越大；整体 α 越小 → 分布越集中
    DIRICHLET_ALPHA = np.array([2.0, 1.0, 1.0, 3.0])  # awgn/rayleigh/rician/multipath

    @staticmethod
    def _loguniform(lo, hi):
        """在 [lo, hi] 上 log-uniform 采样（lo, hi > 0）"""
        return np.exp(np.random.uniform(np.log(lo), np.log(hi)))

    def sample(self, cfg):
        """
        返回一次 batch 使用的物理参数字典。
        所有值均为物理量，尚未换算为采样域。
        """
        p = {}

        # ── 多径 ──────────────────────────────────────────────────────
        # 时延扩展：0.05μs（室内密集）~ 2.0μs（山地/城郊）
        p['tau_rms_us'] = self._loguniform(0.05, 2.0)
        # 抽头数：2（准单径）~ 12（复杂多径），log 域均匀以避免小值欠采样
        p['n_taps'] = max(2, int(round(self._loguniform(2, 12))))

        # ── 莱斯 K 因子 ───────────────────────────────────────────────
        # 0 dB = 纯 NLOS 边界，20 dB = 极强 LOS（室内近场）
        p['rician_k_db'] = np.random.uniform(0.0, 20.0)

        # ── 信道混合权重（Dirichlet）──────────────────────────────────
        # 每次 batch 的 awgn/rayleigh/rician/multipath 占比都不同
        w_raw = np.random.dirichlet(self.DIRICHLET_ALPHA)
        p['mixed_weights'] = dict(zip(
            ['awgn', 'flat_rayleigh', 'flat_rician', 'multipath'], w_raw
        ))

        # ── CFO ───────────────────────────────────────────────────────
        # 先随机选一个精度档，再在 [0, tier] 均匀采样实际频偏上限
        # 这样既有高质量晶振（小 CFO）也有廉价器件（大 CFO）
        tier = float(np.random.choice(self.CFO_TIERS_HZ))
        p['cfo_max_hz'] = np.random.uniform(0.0, tier)

        # ── 相位噪声 PSD ──────────────────────────────────────────────
        # 1e-9 = TCXO/OCXO 级高端，1e-5 = 低成本振荡器
        p['phase_noise_psd'] = self._loguniform(1e-9, 1e-5)

        # ── 时间偏移 ──────────────────────────────────────────────────
        # 两级：先随机偏移上限（0.05~0.5 T_sym），再在其内均匀采
        max_timing = np.random.uniform(0.05, 0.5)
        p['timing_offset_sym'] = max_timing   # 上限本身作为参数传递
                                               # 实际偏移在 _timing_offset 中采样

        # ── IQ 不平衡 ─────────────────────────────────────────────────
        # 两级采样，模拟不同硬件质量
        iq_gain_max = np.random.uniform(0.1, 3.0)    # dB
        iq_phase_max = np.random.uniform(0.5, 10.0)  # deg
        p['iq_gain_imb_max_db']   = iq_gain_max
        p['iq_phase_imb_max_deg'] = iq_phase_max

        # ── DC 偏置 ───────────────────────────────────────────────────
        # log-uniform：低质量前端 DC 偏置可达信号幅度的 5%
        p['dc_offset_max'] = self._loguniform(0.002, 0.05)

        # ── PA（功率放大器）──────────────────────────────────────────
        # IBO：3 dB（严重非线性）~ 15 dB（高线性工作点）
        p['pa_ibo_db'] = np.random.uniform(3.0, 15.0)
        # Rapp 模型阶数：阶数越高截断越陡峭
        p['pa_p'] = int(np.random.choice([1, 2, 3, 4]))

        # ── 多普勒 ────────────────────────────────────────────────────
        # 速度：0（静止）~ 120 km/h（高速公路），混合静止和运动场景
        # 用 log-uniform 但允许 0：以 0.3 的概率置为 0（静止终端）
        if np.random.rand() < 0.3:
            p['doppler_v_kmh'] = 0.0
        else:
            p['doppler_v_kmh'] = self._loguniform(0.5, 120.0)
        # 载波频率：400 MHz（UHF）~ 6000 MHz（Sub-6G）
        p['doppler_fc_mhz'] = self._loguniform(400.0, 6000.0)

        return p


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3. RRC 滤波器                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝
def design_rrc(ntaps, sps, beta):
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
# ║  4. FFT 批量卷积                                                     ║
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
    return Y[:, (K - 1) // 2: (K - 1) // 2 + L]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. 场景固定参数（符号率/采样率相关，不随 batch 变化）                ║
# ╚══════════════════════════════════════════════════════════════════════╝
def compute_scene_fixed(scene_cfg, global_cfg):
    """计算场景固定量：SPS、num_sym、RRC、ADC 位数"""
    sym_rate  = scene_cfg['sym_rate_mhz'] * 1e6
    samp_rate = scene_cfg['samp_rate_mhz'] * 1e6
    sps       = max(1, int(round(samp_rate / sym_rate)))
    num_sym   = SIG_LEN // sps + 10
    rrc       = design_rrc(global_cfg['rrc_taps'], sps, global_cfg['rolloff'])
    thr       = global_cfg['low_rate_threshold_mhz']
    adc_bits  = (global_cfg['adc_bits_low_rate']
                 if global_cfg['enable_quantization_noise']
                    and scene_cfg['samp_rate_mhz'] < thr
                 else global_cfg['adc_bits_high_rate'])
    return {
        'sps': sps, 'num_sym': num_sym, 'rrc': rrc,
        'sym_rate': sym_rate, 'samp_rate': samp_rate,
        'adc_bits': adc_bits,
        'scene_label': f"{scene_cfg['sym_rate_mhz']}M/{scene_cfg['samp_rate_mhz']}M",
    }


def derive_sampled_params(phys, sf):
    """
    将本次 batch 采样到的物理参数（phys）换算为采样域参数，
    结合场景固定量（sf: scene_fixed）。
    返回 sp 字典，供信道/损伤模型直接使用。
    """
    fs  = sf['samp_rate']   # Hz
    sps = sf['sps']

    sp = dict(sf)  # 复制场景固定量

    # 多径
    sp['tau_rms_samp'] = max(phys['tau_rms_us'] * 1e-6 * fs, 0.5)
    sp['n_taps']       = phys['n_taps']

    # 莱斯 K
    sp['rician_k_db']  = phys['rician_k_db']

    # 信道权重（本 batch 采样）
    sp['mixed_weights'] = phys['mixed_weights']

    # CFO
    sp['cfo_norm']      = phys['cfo_max_hz'] / fs

    # 相位噪声：σ/sample = sqrt(PSD × fs)
    sp['phase_noise_std'] = np.sqrt(phys['phase_noise_psd'] * fs)

    # 时间偏移（上限换算为样本数）
    sp['timing_off_samp'] = phys['timing_offset_sym'] * sps

    # IQ 不平衡（已是物理单位，直接传递）
    sp['iq_gain_imb_max_db']   = phys['iq_gain_imb_max_db']
    sp['iq_phase_imb_max_deg'] = phys['iq_phase_imb_max_deg']

    # DC
    sp['dc_offset_max'] = phys['dc_offset_max']

    # PA
    sp['pa_ibo_db'] = phys['pa_ibo_db']
    sp['pa_p']      = phys['pa_p']

    # 多普勒
    if phys['doppler_v_kmh'] > 0:
        v_ms     = phys['doppler_v_kmh'] / 3.6
        fc_hz    = phys['doppler_fc_mhz'] * 1e6
        fd_max   = v_ms / 3e8 * fc_hz
        sp['doppler_norm'] = fd_max / fs
    else:
        sp['doppler_norm'] = 0.0

    return sp


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. 星座定义                                                         ║
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
# ║  7. 调制信号生成器                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _rand_symbols(name, n_batch, n_sym):
    c = CONSTELLATIONS[name]
    return c[np.random.randint(0, len(c), (n_batch, n_sym))]

def _gen_linear(const_name, n, sp):
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    syms = _rand_symbols(const_name, n, num_sym)
    up   = np.zeros((n, num_sym * sps), dtype=np.complex128)
    up[:, ::sps] = syms
    return batch_fft_convolve(up, rrc)[:, :SIG_LEN]

def gen_bpsk(n, sp):   return _gen_linear('BPSK', n, sp)
def gen_qpsk(n, sp):   return _gen_linear('QPSK', n, sp)
def gen_16qam(n, sp):  return _gen_linear('16QAM', n, sp)
def gen_64qam(n, sp):  return _gen_linear('64QAM', n, sp)

def gen_oqpsk(n, sp):
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    bi = 2.0 * np.random.randint(0, 2, (n, num_sym)) - 1.0
    bq = 2.0 * np.random.randint(0, 2, (n, num_sym)) - 1.0
    L  = num_sym * sps
    up_i = np.zeros((n, L)); up_i[:, ::sps] = bi
    up_q = np.zeros((n, L))
    q_idx = np.arange(num_sym) * sps + sps // 2
    q_idx = q_idx[q_idx < L]
    up_q[:, q_idx] = bq[:, :len(q_idx)]
    si = batch_fft_convolve(up_i, rrc).real
    sq = batch_fft_convolve(up_q, rrc).real
    return (si + 1j * sq)[:, :SIG_LEN]

def gen_pi4dqpsk(n, sp):
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    dphi  = np.array([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4])
    phase = np.cumsum(dphi[np.random.randint(0, 4, (n, num_sym))], axis=1)
    up    = np.zeros((n, num_sym * sps), dtype=np.complex128)
    up[:, ::sps] = np.exp(1j * phase)
    return batch_fft_convolve(up, rrc)[:, :SIG_LEN]

def _gen_cpfsk(n, sp, M, h_mod):
    sps, num_sym = sp['sps'], sp['num_sym']
    levels   = 2.0 * np.arange(M) - (M - 1)
    freq_sym = levels[np.random.randint(0, M, (n, num_sym))]
    freq_up  = np.repeat(h_mod * freq_sym / (2.0 * sps), sps, axis=1)[:, :SIG_LEN]
    return np.exp(1j * 2 * np.pi * np.cumsum(freq_up, axis=1))

def gen_2fsk(n, sp):  return _gen_cpfsk(n, sp, 2, 1.0)
# def gen_4fsk(n, sp):  return _gen_cpfsk(n, sp, 4, 1.0/3.0)
def gen_4fsk(n, sp):  return _gen_cpfsk(n, sp, 4, 1.0/3.0)

def gen_gmsk(n, sp, BT=0.3):
    sps, num_sym = sp['sps'], sp['num_sym']
    # h_mod=0.5: 每样本频偏 = h_mod × (±1) / (2 × sps)
    # 每符号累积相位 = 2π × freq_per_sample × sps = ±π/2，保证 h=0.5
    # 原 dynamic 版写成 ±0.5/sps，缺少分母中的 2，导致 h=1.0，信号退化为类 BPSK
    freq_sym = np.where(np.random.randint(0, 2, (n, num_sym)) == 0, -1.0, 1.0)
    freq_rect = np.repeat(0.5 * freq_sym / (2.0 * sps), sps, axis=1)
    B = BT
    t_g = np.arange(-2*sps, 2*sps+1)
    h_g = np.exp(-2*(np.pi * B * t_g / sps)**2 / np.log(2))
    h_g /= h_g.sum()
    freq_up = batch_fft_convolve(freq_rect, h_g).real[:, :SIG_LEN]
    return np.exp(1j * 2 * np.pi * np.cumsum(freq_up, axis=1))

GENERATORS = {
    'BPSK': gen_bpsk, 'QPSK': gen_qpsk, 'OQPSK': gen_oqpsk,
    '2FSK': gen_2fsk, '4FSK': gen_4fsk,
    '16QAM': gen_16qam, '64QAM': gen_64qam,
    'Pi4DQPSK': gen_pi4dqpsk, 'GMSK': gen_gmsk,
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  8. 量化噪声                                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝
def apply_quantization(x, n_bits):
    if n_bits <= 0:
        return x
    m = np.max(np.abs(x))
    if m < 1e-12:
        return x
    xn   = x / m
    step = 2.0 / (2 ** n_bits)
    Iq   = np.clip(np.round(xn.real / step) * step, -1.0, 1.0)
    Qq   = np.clip(np.round(xn.imag / step) * step, -1.0, 1.0)
    return (Iq + 1j * Qq) * m


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  9. 信道模型（使用 sp 中换算后的采样域参数）                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
class ChannelModel:

    @staticmethod
    def _norm(x):
        pwr = np.mean(np.abs(x)**2, axis=-1, keepdims=True)
        return x / np.sqrt(pwr + 1e-12)

    def _flat_rayleigh(self, x, sp):
        N = x.shape[0]
        h0 = (np.random.randn(N, 1) + 1j*np.random.randn(N, 1)) / np.sqrt(2)
        if sp['doppler_norm'] > 0:
            fd   = np.random.uniform(0, sp['doppler_norm'], (N, 1))
            phi0 = np.random.uniform(0, 2*np.pi, (N, 1))
            n    = np.arange(x.shape[1]).reshape(1, -1)
            h0   = h0 * np.exp(1j * (2*np.pi*fd*n + phi0))
        return x * h0

    def _flat_rician(self, x, sp):
        N = x.shape[0]
        K = 10 ** (sp['rician_k_db'] / 10)
        # LOS 多普勒（随机入射角）
        phi_los = np.random.uniform(0, 2*np.pi, (N, 1))
        h_nlos  = (np.random.randn(N, 1) + 1j*np.random.randn(N, 1)) / np.sqrt(2*(K+1))
        if sp['doppler_norm'] > 0:
            theta = np.random.uniform(0, 2*np.pi, (N, 1))
            fd_los = sp['doppler_norm'] * np.cos(theta)
            n      = np.arange(x.shape[1]).reshape(1, -1)
            h_los  = np.sqrt(K/(K+1)) * np.exp(1j*(2*np.pi*fd_los*n + phi_los))
            fd_nl  = np.random.uniform(-sp['doppler_norm'], sp['doppler_norm'], (N, 1))
            h_nlos = h_nlos * np.exp(1j*2*np.pi*fd_nl*n)
        else:
            h_los = np.sqrt(K/(K+1)) * np.exp(1j*phi_los)
        return x * (h_los + h_nlos)

    def _multipath(self, x, sp):
        N, L   = x.shape
        n_taps = sp['n_taps']
        tau    = sp['tau_rms_samp']
        pdp    = np.exp(-np.arange(n_taps, dtype=np.float64) / tau)
        pdp   /= pdp.sum()
        h = (np.random.randn(N, n_taps) + 1j*np.random.randn(N, n_taps)) * \
            np.sqrt(pdp[np.newaxis, :] / 2)
        y = np.zeros_like(x)
        for k in range(n_taps):
            if k == 0:
                y += h[:, 0:1] * x
            else:
                y[:, k:] += h[:, k:k+1] * x[:, :-k]
        return y

    @staticmethod
    def _awgn(x, snr_db):
        sigma2 = 10.0 ** (-snr_db / 10.0)
        return x + np.sqrt(sigma2/2) * (
            np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape))

    def apply(self, signals, snr_db, sp):
        x  = self._norm(signals)
        w  = sp['mixed_weights']
        N  = x.shape[0]
        types = list(w.keys())
        probs = [w[t] for t in types]
        assign = np.random.choice(types, size=N, p=probs)
        y = x.copy()
        for ch in types:
            m = assign == ch
            if not m.any():
                continue
            sub = x[m]
            if ch == 'flat_rayleigh':
                sub = self._flat_rayleigh(sub, sp)
            elif ch == 'flat_rician':
                sub = self._flat_rician(sub, sp)
            elif ch == 'multipath':
                sub = self._multipath(sub, sp)
            y[m] = sub
        return self._awgn(y, snr_db).astype(np.complex64)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  10. RF 损伤（使用 sp 中换算后的采样域参数）                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
class RFImpairments:

    def apply_tx(self, x, sp):
        return self._pa_rapp(x, sp['pa_ibo_db'], sp['pa_p'])

    @staticmethod
    def _pa_rapp(x, ibo_db, p):
        pwr   = np.mean(np.abs(x)**2, axis=-1, keepdims=True)
        A_sat = np.sqrt(pwr * 10**(ibo_db/10))
        r     = np.abs(x)
        r_out = r / (1 + (r/(A_sat+1e-12))**(2*p)) ** (1/(2*p))
        return r_out * np.exp(1j * np.angle(x))

    def apply_rx(self, x, sp, cfg):
        if not cfg['impairments_enable']:
            return x
        x = self._cfo(x, sp)
        x = self._phase_noise(x, sp)
        x = self._iq_imbalance(x, sp)
        x = self._dc_offset(x, sp)
        x = self._timing_offset(x, sp)
        return x

    @staticmethod
    def _cfo(x, sp):
        N, L   = x.shape
        max_fn = sp['cfo_norm']
        if max_fn <= 0:
            return x
        df  = np.random.uniform(-max_fn, max_fn, (N, 1))
        phi = np.random.uniform(0, 2*np.pi, (N, 1))
        n   = np.arange(L).reshape(1, L)
        return x * np.exp(1j * (2*np.pi*df*n + phi))

    @staticmethod
    def _phase_noise(x, sp):
        N, L  = x.shape
        sigma = sp['phase_noise_std']
        if sigma <= 0:
            return x
        pn = np.cumsum(sigma * np.random.randn(N, L), axis=1)
        return x * np.exp(1j * pn)

    @staticmethod
    def _iq_imbalance(x, sp):
        N     = x.shape[0]
        g_max = sp['iq_gain_imb_max_db']
        p_max = sp['iq_phase_imb_max_deg']
        eps   = 10**(np.random.uniform(-g_max, g_max, (N, 1))/20) - 1
        dphi  = np.random.uniform(-p_max, p_max, (N, 1)) * np.pi/180
        I, Q  = x.real, x.imag
        return (1+eps/2)*I + 1j*(1-eps/2)*(Q*np.cos(dphi)+I*np.sin(dphi))

    @staticmethod
    def _dc_offset(x, sp):
        N  = x.shape[0]
        mx = sp['dc_offset_max']
        dc = np.random.uniform(-mx, mx, (N, 1)) + \
             1j*np.random.uniform(-mx, mx, (N, 1))
        return x + dc

    @staticmethod
    def _timing_offset(x, sp):
        N, L    = x.shape
        mx      = sp['timing_off_samp']
        if mx <= 0:
            return x
        offsets = np.random.uniform(-mx, mx, N)
        t  = np.arange(L, dtype=np.float64)
        y  = np.empty_like(x)
        for i in range(N):
            tn   = np.clip(t + offsets[i], 0, L-1)
            y[i] = np.interp(tn, t, x[i].real) + 1j*np.interp(tn, t, x[i].imag)
        return y


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  11. 主生成流水线                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
def generate_all(cfg=CONFIG):
    np.random.seed(cfg['seed'])

    sampler     = PhysicalParamSampler()
    channel     = ChannelModel()
    impairments = RFImpairments()

    classes  = cfg['classes']
    snr_list = cfg['snr_list_db']
    n_per    = cfg['n_per_class_per_snr']
    scenes   = cfg['symbol_rates']

    # 计算各场景固定量（SPS / RRC / ADC，不随 batch 变化）
    scene_fixed_list = [compute_scene_fixed(s, cfg) for s in scenes]

    print("\n  场景固定量:")
    for sf in scene_fixed_list:
        print(f"    {sf['scene_label']:25s}  SPS={sf['sps']}  "
              f"fs={sf['samp_rate']/1e6:.4f} MHz  ADC={sf['adc_bits']}bit")
    print()

    all_iq, all_y, all_snr, all_sr, all_scene = [], [], [], [], []
    total = len(scenes) * len(classes) * len(snr_list)
    pbar  = tqdm(total=total, desc='Generating', ncols=95)

    for sc_idx, (scene_cfg, sf) in enumerate(zip(scenes, scene_fixed_list)):
        for cls_name in classes:
            gen_fn = GENERATORS[cls_name]

            for snr_db in snr_list:
                # ① 本 batch 独立采样物理参数
                phys = sampler.sample(cfg)

                # ② 物理参数 → 采样域参数（换算）
                sp = derive_sampled_params(phys, sf)

                # ③ 生成基带信号
                clean = gen_fn(n_per, sp)

                # ④ TX 损伤（PA，使用本 batch 的 IBO/p）
                if cfg['pa_enable']:
                    clean = impairments.apply_tx(clean, sp)

                # ⑤ 信道（衰落+多普勒+AWGN）
                noisy = channel.apply(clean, snr_db, sp)

                # ⑥ 量化噪声
                if cfg['enable_quantization_noise'] and sf['adc_bits'] < 12:
                    noisy = apply_quantization(noisy, sf['adc_bits']).astype(np.complex64)

                # ⑦ RX 损伤（CFO/相噪/IQ/DC/时偏）
                noisy = impairments.apply_rx(noisy, sp, cfg)

                # ⑧ 转 IQ: (N, 2, SIG_LEN)
                iq = np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)

                all_iq.append(iq)
                cls_idx = classes.index(cls_name)
                all_y.extend([cls_idx] * n_per)
                all_snr.extend([float(snr_db)] * n_per)
                all_sr.extend([sf['sym_rate']] * n_per)
                all_scene.extend([sc_idx] * n_per)

                pbar.set_postfix(scene=sf['scene_label'], cls=cls_name,
                                 snr=f'{snr_db:+d}dB',
                                 τ=f"{phys['tau_rms_us']:.2f}μs",
                                 v=f"{phys['doppler_v_kmh']:.0f}kmh")
                pbar.update(1)

    pbar.close()

    X     = np.concatenate(all_iq, axis=0)
    Y     = np.array(all_y,    dtype=np.int64)
    SNR   = np.array(all_snr,  dtype=np.float32)
    SR    = np.array(all_sr,   dtype=np.float32)
    SCENE = np.array(all_scene, dtype=np.int64)
    print(f'\n  总样本: {len(Y):,}   X shape: {X.shape}')
    return X, Y, SNR, SR, SCENE


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  12. 保存                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
def split_and_save(X, Y, SNR, SR, SCENE, cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    idx = np.arange(len(Y))
    i_tr, i_te = train_test_split(
        idx, test_size=1-cfg['train_ratio'],
        random_state=cfg['seed'], stratify=Y
    )
    scene_labels = [
        f"{s['sym_rate_mhz']}M_sym_{s['samp_rate_mhz']}M_samp"
        for s in cfg['symbol_rates']
    ]
    path = os.path.join(cfg['save_dir'], 'amr_multirate_dynamic_4.npz')
    np.savez_compressed(
        path,
        X_train=X[i_tr], Y_train=Y[i_tr],
        X_test=X[i_te],  Y_test=Y[i_te],
        snr_train=SNR[i_tr], snr_test=SNR[i_te],
        sr_train=SR[i_tr],   sr_test=SR[i_te],
        scene_train=SCENE[i_tr], scene_test=SCENE[i_te],
        classes=np.array(cfg['classes']),
        snr_list=np.array(cfg['snr_list_db'], dtype=np.float32),
        scene_labels=np.array(scene_labels),
    )
    sz = os.path.getsize(path) / 1024**2
    print(f'  已保存 → {path}')
    print(f'    Train: {len(i_tr):,}  Test: {len(i_te):,}  Size: {sz:.1f} MB')
    return path


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  13. MAIN                                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    n_scenes = len(CONFIG['symbol_rates'])
    n_total  = n_scenes * NUM_CLS * len(CONFIG['snr_list_db']) * CONFIG['n_per_class_per_snr']

    print('=' * 65)
    print('  Multi-Rate AMR Dataset Generator [Dynamic Physical Params]')
    print(f'  Classes: {NUM_CLS}   Scenes: {n_scenes}   Total: {n_total:,}')
    print(f'  SNR: {CONFIG["snr_list_db"][0]}~{CONFIG["snr_list_db"][-1]} dB '
          f'({len(CONFIG["snr_list_db"])} steps)')
    print('─' * 65)
    print('  每个 batch 独立采样的物理参数范围:')
    print('    τ_rms       [0.05, 2.0] μs        log-uniform')
    print('    n_taps      [2, 12]                log-uniform (整数)')
    print('    K_rician    [0, 20] dB             uniform')
    print('    CFO         {100,500,2k,8k} Hz     分类→uniform 两级')
    print('    相位噪声PSD  [1e-9, 1e-5] rad²/Hz   log-uniform')
    print('    时间偏移上限  [0.05, 0.5] T_sym      uniform→sub')
    print('    IQ增益不平衡 [0.1, 3.0] dB          uniform→sub')
    print('    IQ相位不平衡 [0.5, 10.0] deg         uniform→sub')
    print('    DC偏置       [0.002, 0.05]          log-uniform')
    print('    PA IBO       [3, 15] dB             uniform')
    print('    PA p阶数     {1,2,3,4}              离散均匀')
    print('    速度         [0, 120] km/h          30%=0, 70%=log-uniform')
    print('    载波频率     [400, 6000] MHz         log-uniform')
    print('    信道权重     Dirichlet(α=[2,1,1,3])  Dirichlet')
    print('=' * 65)

    X, Y, SNR, SR, SCENE = generate_all()
    split_and_save(X, Y, SNR, SR, SCENE)
    print('\n✓ Done!')
