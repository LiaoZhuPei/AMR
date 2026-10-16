# -*- coding: utf-8 -*-
"""
generator_denoise.py
修改后的多速率、带物理对齐去噪对的数据集生成程序
"""
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 显式使用非交互式后端
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONFIG = {
    'signal_length': 1024,
    'symbol_rates': [
        {'sym_rate_mhz': 0.7812, 'samp_rate_mhz': 3.125},
        {'sym_rate_mhz': 1.5625, 'samp_rate_mhz': 6.25},
        {'sym_rate_mhz': 3.125, 'samp_rate_mhz': 12.5},
    ],
    'n_per_class_per_snr': 8,
    'snr_list_db': list(range(-2, 38, 2)),
    'rolloff': 0.35,
    'rrc_taps': 101,
    'train_ratio': 0.8,
    'save_dir': './multirate_dataset',
    'channel_type': 'mixed',
    'mixed_weights': {
        'awgn': 0.30,
        'flat_rayleigh': 0.20,
        'flat_rician': 0.15,
        'multipath': 0.35,
    },
    'rician_k_db': 10.0,
    'multipath_taps': 8,
    'multipath_tau_rms': 3.0,
    'pa_enable': True,
    'pa_ibo_db': 8.0,
    'pa_p': 3,
    'impairments_enable': True,
    'cfo_max_norm': 0.001,
    'phase_noise_std': 0.010,
    'iq_gain_imb_max_db': 1.0,
    'iq_phase_imb_max_deg': 5.0,
    'dc_offset_max': 0.020,
    'timing_offset_max': 0.4,
    'enable_quantization_noise': True,
    'adc_bits_high_rate': 12,
    'adc_bits_low_rate': 8,
    'low_rate_threshold_mhz': 0.5,
    'classes': ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM', '128QAM', '256QAM', 'MSK'],
    'seed': 42,
}

SIG_LEN = CONFIG['signal_length']
NUM_CLS = len(CONFIG['classes'])


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


def compute_scene_params(scene_cfg, global_cfg):
    sym_rate = scene_cfg['sym_rate_mhz'] * 1e6
    samp_rate = scene_cfg['samp_rate_mhz'] * 1e6
    sps_float = samp_rate / sym_rate
    sps_int = max(1, int(round(sps_float)))
    num_sym = SIG_LEN // sps_int + 10
    rrc = design_rrc(global_cfg['rrc_taps'], sps_int, global_cfg['rolloff'])
    thr = global_cfg['low_rate_threshold_mhz']
    if global_cfg['enable_quantization_noise'] and scene_cfg['samp_rate_mhz'] < thr:
        adc_bits = global_cfg['adc_bits_low_rate']
    else:
        adc_bits = global_cfg['adc_bits_high_rate']
    return {
        'sps': sps_int, 'sps_float': sps_float, 'num_sym': num_sym,
        'rrc': rrc, 'sym_rate': sym_rate, 'samp_rate': samp_rate,
        'adc_bits': adc_bits, 'scene_label': f"{scene_cfg['sym_rate_mhz']}M_{scene_cfg['samp_rate_mhz']}M",
    }


def _make_psk(M):
    return np.exp(1j * 2 * np.pi * np.arange(M) / M)


def _make_square_qam(M):
    k = int(np.sqrt(M))
    levels = 2.0 * np.arange(k) - (k - 1)
    gI, gQ = np.meshgrid(levels, levels)
    pts = (gI + 1j * gQ).flatten()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


# ╔══════════════════════════════════════════════════════════════╗
# ║ 【关键修改】标准、对齐硬件定义的 32/128QAM 偶数网格生成逻辑 ║
# ╚══════════════════════════════════════════════════════════════╝
def _make_cross_qam(M):
    if M == 32:
        # 标准 32QAM: 6x6 偶数网格 (36点)，切掉 4 个角上各 1 个点 (36 - 4 = 32)
        levels = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float64)
        gI, gQ = np.meshgrid(levels, levels)
        pts = (gI + 1j * gQ).flatten()
        mask = ~((np.abs(pts.real) == 5) & (np.abs(pts.imag) == 5))
        pts = pts[mask]
    elif M == 128:
        # 标准 128QAM: 12x12 偶数网格 (144点)，切掉 4 个角上各 2x2 = 4 个点 (144 - 16 = 128)
        levels = np.arange(-11, 12, 2, dtype=np.float64)
        gI, gQ = np.meshgrid(levels, levels)
        pts = (gI + 1j * gQ).flatten()
        mask = ~((np.abs(pts.real) >= 9) & (np.abs(pts.imag) >= 9))
        pts = pts[mask]
    else:
        # 兜底原逻辑
        side = int(np.ceil(np.sqrt(M)))
        if side % 2 == 0:
            side += 1
        levels = 2.0 * np.arange(side) - (side - 1)
        gI, gQ = np.meshgrid(levels, levels)
        pts = (gI + 1j * gQ).flatten()
        idx = np.argsort(np.abs(pts) ** 2)[:M]
        pts = pts[idx]
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS = {
    'BPSK': _make_psk(2),
    'QPSK': _make_psk(4),
    '8PSK': _make_psk(8),
    '16QAM': _make_square_qam(16),
    '32QAM': _make_cross_qam(32),
    '64QAM': _make_square_qam(64),
    '128QAM': _make_cross_qam(128),
    '256QAM': _make_square_qam(256),
}


def _rand_symbols(name, n_batch, n_sym):
    c = CONSTELLATIONS[name]
    return c[np.random.randint(0, len(c), (n_batch, n_sym))]


def _gen_linear(const_name, n, sp):
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    syms = _rand_symbols(const_name, n, num_sym)
    up_len = num_sym * sps
    up = np.zeros((n, up_len), dtype=np.complex128)
    up[:, ::sps] = syms
    out = batch_fft_convolve(up, rrc)
    return out[:, :SIG_LEN]


def gen_bpsk(n, sp):    return _gen_linear('BPSK', n, sp)


def gen_qpsk(n, sp):    return _gen_linear('QPSK', n, sp)


def gen_8psk(n, sp):    return _gen_linear('8PSK', n, sp)


def gen_16qam(n, sp):   return _gen_linear('16QAM', n, sp)


def gen_32qam(n, sp):   return _gen_linear('32QAM', n, sp)


def gen_64qam(n, sp):   return _gen_linear('64QAM', n, sp)


def gen_128qam(n, sp):  return _gen_linear('128QAM', n, sp)


def gen_256qam(n, sp):  return _gen_linear('256QAM', n, sp)


def gen_2ask(n, sp):
    sps, num_sym, rrc = sp['sps'], sp['num_sym'], sp['rrc']
    syms = np.random.randint(0, 2, (n, num_sym)).astype(np.float64) * np.sqrt(2)
    up_len = num_sym * sps
    up = np.zeros((n, up_len), dtype=np.float64)
    up[:, ::sps] = syms
    out = batch_fft_convolve(up, rrc).real
    return out[:, :SIG_LEN].astype(np.complex128)


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
    sps, num_sym = sp['sps'], sp['num_sym']
    syms = np.random.randint(0, M, (n, num_sym))
    levels = 2.0 * np.arange(M) - (M - 1)
    freq_sym = levels[syms]
    freq_per_sample = h_mod * freq_sym / (2.0 * sps)
    freq_up = np.repeat(freq_per_sample, sps, axis=1)[:, :SIG_LEN]
    phase = 2 * np.pi * np.cumsum(freq_up, axis=1)
    return np.exp(1j * phase)


def gen_2fsk(n, sp):   return _gen_cpfsk(n, sp, M=2, h_mod=1.0)


def gen_4fsk(n, sp):   return _gen_cpfsk(n, sp, M=4, h_mod=1.0 / 3.0)


def gen_msk(n, sp):    return _gen_cpfsk(n, sp, M=2, h_mod=0.5)


GENERATORS = {
    'BPSK': gen_bpsk, 'QPSK': gen_qpsk, '8PSK': gen_8psk, 'OQPSK': gen_oqpsk,
    '2FSK': gen_2fsk, '4FSK': gen_4fsk, '16QAM': gen_16qam, '2ASK': gen_2ask,
    'Pi4DQPSK': gen_pi4dqpsk, '64QAM': gen_64qam, '32QAM': gen_32qam,
    '128QAM': gen_128qam, '256QAM': gen_256qam, 'MSK': gen_msk,
}


def apply_quantization(x, n_bits):
    if n_bits <= 0: return x
    max_val = np.max(np.abs(x))
    if max_val < 1e-12: return x
    x_norm = x / max_val
    levels = 2 ** n_bits
    step = 2.0 / levels
    I_q = np.round(x_norm.real / step) * step
    Q_q = np.round(x_norm.imag / step) * step
    I_q = np.clip(I_q, -1.0, 1.0)
    Q_q = np.clip(Q_q, -1.0, 1.0)
    return (I_q + 1j * Q_q) * max_val


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
        h_los = np.sqrt(K / (K + 1)) * np.exp(1j * phi)
        h_nlos = np.sqrt(1 / (2 * (K + 1))) * (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
        return x * (h_los + h_nlos)

    def _multipath(self, x):
        N, L = x.shape
        n_taps = self.cfg['multipath_taps']
        tau_rms = self.cfg['multipath_tau_rms']
        pdp = np.exp(-np.arange(n_taps, dtype=np.float64) / tau_rms)
        pdp /= pdp.sum()
        h = np.zeros((N, n_taps), dtype=np.complex128)
        for k in range(n_taps):
            h[:, k] = np.sqrt(pdp[k] / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
        y = np.zeros_like(x)
        for k in range(n_taps):
            if k == 0:
                y += h[:, 0:1] * x
            else:
                y[:, k:] += h[:, k:k + 1] * x[:, :-k]
        return y

    def _awgn(self, x, snr_db):
        sigma2 = 10.0 ** (-snr_db / 10.0)
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
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
            if not m.any(): continue
            sub = x[m]
            if ch == 'flat_rayleigh':
                sub = self._flat_rayleigh(sub)
            elif ch == 'flat_rician':
                sub = self._flat_rician(sub)
            elif ch == 'multipath':
                sub = self._multipath(sub)
            y[m] = sub
        return y


class RFImpairments:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply_tx(self, x):
        if self.cfg['pa_enable']: x = self._pa_rapp(x)
        return x

    def _pa_rapp(self, x):
        p = self.cfg['pa_p']
        ibo = self.cfg['pa_ibo_db']
        pwr = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
        A_sat = np.sqrt(pwr * 10 ** (ibo / 10))
        r = np.abs(x)
        theta = np.angle(x)
        r_out = r / (1 + (r / (A_sat + 1e-12)) ** (2 * p)) ** (1 / (2 * p))
        return r_out * np.exp(1j * theta)

    def apply_rx(self, x):
        if not self.cfg['impairments_enable']: return x
        x = self._cfo(x)
        x = self._phase_noise(x)
        x = self._iq_imbalance(x)
        x = self._dc_offset(x)
        x = self._timing_offset(x)
        return x

    def _cfo(self, x):
        N, L = x.shape
        max_cfo = self.cfg['cfo_max_norm']
        if max_cfo <= 0: return x
        df = np.random.uniform(-max_cfo, max_cfo, (N, 1))
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        n = np.arange(L).reshape(1, L)
        return x * np.exp(1j * (2 * np.pi * df * n + phi))

    def _phase_noise(self, x):
        N, L = x.shape
        sigma = self.cfg['phase_noise_std']
        if sigma <= 0: return x
        increments = sigma * np.random.randn(N, L)
        pn = np.cumsum(increments, axis=1)
        return x * np.exp(1j * pn)

    def _iq_imbalance(self, x):
        N, L = x.shape
        g_max = self.cfg['iq_gain_imb_max_db']
        p_max = self.cfg['iq_phase_imb_max_deg']
        if g_max <= 0 and p_max <= 0: return x
        eps = 10 ** (np.random.uniform(-g_max, g_max, (N, 1)) / 20) - 1
        dphi = np.random.uniform(-p_max, p_max, (N, 1)) * np.pi / 180
        I, Q = x.real, x.imag
        I_out = (1 + eps / 2) * I
        Q_out = (1 - eps / 2) * (Q * np.cos(dphi) + I * np.sin(dphi))
        return I_out + 1j * Q_out

    def _dc_offset(self, x):
        N, L = x.shape
        mx = self.cfg['dc_offset_max']
        if mx <= 0: return x
        dc = np.random.uniform(-mx, mx, (N, 1)) + 1j * np.random.uniform(-mx, mx, (N, 1))
        return x + dc

    def _timing_offset(self, x):
        N, L = x.shape
        mx = self.cfg['timing_offset_max']
        if mx <= 0: return x
        offsets = np.random.uniform(-mx, mx, N)
        t = np.arange(L, dtype=np.float64)
        y = np.empty_like(x)
        for i in range(N):
            t_new = np.clip(t + offsets[i], 0, L - 1)
            y[i] = np.interp(t_new, t, x[i].real) + 1j * np.interp(t_new, t, x[i].imag)
        return y


# ╔══════════════════════════════════════════════════════════════╗
# ║  【核心修改】同时提取理想无噪与畸变带噪两路信号              ║
# ╚══════════════════════════════════════════════════════════════╝
def generate_all(cfg=CONFIG):
    np.random.seed(cfg['seed'])
    channel = ChannelModel(cfg)
    impairments = RFImpairments(cfg)

    classes = cfg['classes']
    snr_list = cfg['snr_list_db']
    n_per = cfg['n_per_class_per_snr']
    scenes = cfg['symbol_rates']
    scene_params = [compute_scene_params(s, cfg) for s in scenes]

    all_iq_noisy, all_iq_clean = [], []
    all_y, all_snr, all_sr, all_scene = [], [], [], []

    total = len(scenes) * len(classes) * len(snr_list)
    pbar = tqdm(total=total, desc='Generating Pairs', ncols=95)

    for sc_idx, (scene_cfg, sp) in enumerate(zip(scenes, scene_params)):
        for cls_idx, cls_name in enumerate(classes):
            gen_fn = GENERATORS[cls_name]
            for snr_db in snr_list:
                # 1. 基础信号生成 (这就是最理想、零失真的发射基带波形)
                clean = gen_fn(n_per, sp)

                # 理想基带平均功率归一化
                pwr_clean = np.mean(np.abs(clean) ** 2, axis=-1, keepdims=True)
                clean_norm = clean / np.sqrt(pwr_clean + 1e-12)
                iq_clean = np.stack([clean_norm.real, clean_norm.imag], axis=1).astype(np.float32)

                # 2. 依次叠加发射损伤、信道损耗、接收损伤，最终退化为带噪畸变信号
                noisy = impairments.apply_tx(clean)
                noisy = channel.apply(noisy, snr_db)
                if cfg['enable_quantization_noise'] and sp['adc_bits'] < 12:
                    noisy = apply_quantization(noisy, sp['adc_bits']).astype(np.complex64)
                noisy = impairments.apply_rx(noisy)

                # 带噪信号平均功率归一化 (与你的分类网络 DataLoader 中的 power_normalize 完全一致)
                pwr_noisy = np.mean(np.abs(noisy) ** 2, axis=-1, keepdims=True)
                noisy_norm = noisy / np.sqrt(pwr_noisy + 1e-12)
                iq_noisy = np.stack([noisy_norm.real, noisy_norm.imag], axis=1).astype(np.float32)

                all_iq_noisy.append(iq_noisy)
                all_iq_clean.append(iq_clean)

                all_y.extend([cls_idx] * n_per)
                all_snr.extend([float(snr_db)] * n_per)
                all_sr.extend([sp['sym_rate']] * n_per)
                all_scene.extend([sc_idx] * n_per)

                pbar.update(1)
    pbar.close()

    X_noisy = np.concatenate(all_iq_noisy, axis=0)
    X_clean = np.concatenate(all_iq_clean, axis=0)
    Y = np.array(all_y, dtype=np.int64)
    SNR = np.array(all_snr, dtype=np.float32)
    SR = np.array(all_sr, dtype=np.float32)
    SCENE = np.array(all_scene, dtype=np.int64)

    return X_noisy, X_clean, Y, SNR, SR, SCENE


def split_and_save(X_noisy, X_clean, Y, SNR, SR, SCENE, cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    idx = np.arange(len(Y))
    i_tr, i_te = train_test_split(
        idx, test_size=1 - cfg['train_ratio'],
        random_state=cfg['seed'], stratify=Y
    )

    scene_labels = [f"{s['sym_rate_mhz']}M_sym_{s['samp_rate_mhz']}M_samp" for s in cfg['symbol_rates']]
    path = os.path.join(cfg['save_dir'], 'amr_multirate_1.npz')

    # ╔══════════════════════════════════════════════════════════════╗
    # ║ 【关键修改】向原分类的 .npz 存档追加写入 X_clean 纯净去噪对  ║
    # ╚══════════════════════════════════════════════════════════════╝
    np.savez_compressed(
        path,
        X_train=X_noisy[i_tr],
        X_clean_train=X_clean[i_tr],  # 新增：去噪理想训练对
        Y_train=Y[i_tr],
        X_test=X_noisy[i_te],
        X_clean_test=X_clean[i_te],  # 新增：去噪理想测试对
        Y_test=Y[i_te],
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
    print(f'  Dataset Saved successfully → {path}')
    print(f'    Train Pairs: {len(i_tr):,}  Test Pairs: {len(i_te):,}  Size: {sz:.1f} MB')
    return path


if __name__ == '__main__':
    print('=' * 65)
    print('  Multi-Rate AMR Dataset Generator WITH CLEAN-PAIR TARGET')
    print('=' * 65)
    X_n, X_c, Y, SNR, SR, SC = generate_all()
    split_and_save(X_n, X_c, Y, SNR, SR, SC)