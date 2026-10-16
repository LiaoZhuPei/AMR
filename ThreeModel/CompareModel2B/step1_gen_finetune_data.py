"""
step1_gen_finetune_data.py
生成微调数据集：增益改为对数均匀分布，覆盖完整范围 [0.005, 0.5]
低增益区间样本密度大幅提升，同时高增益样本依然存在
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  配置（只改了标注★的参数，其余与原始训练完全一致）                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    # ---- 信号参数（与原训练完全一致，不要改） ----
    'signal_length':        1024,
    'sps':                  4,
    'rolloff':              0.35,
    'rrc_taps':             101,

    # ---- ★ 数据集规模（缩小以加快速度）----
    'n_per_class_per_snr':  100,                       # 原来400，缩小4倍
    'snr_list_db':          list(range(-2, 30, 2)),    # 17档，原来40档
    'train_ratio':          0.8,
    'save_dir':             './finetune_v2_dataset',

    # ---- 信道（与原训练完全一致）----
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

    # ---- 发射端损伤（与原训练完全一致）----
    'pa_enable':            True,
    'pa_ibo_db':            8.0,
    'pa_p':                 3,

    # ---- 接收端损伤（与原训练完全一致）----
    'impairments_enable':   True,
    'cfo_max_norm':         0.002,
    'phase_noise_std':      0.015,
    'iq_gain_imb_max_db':   1.5,
    'iq_phase_imb_max_deg': 5.0,
    'dc_offset_max':        0.020,
    'timing_offset_max':    0.4,

    # ---- ADC（与原训练完全一致）----
    'adc_bit_depth':        16,
    # ★ 关键：范围不缩窄，保持原始完整范围
    'gain_range':           [0.005, 0.5],

    # ---- 调制类别（全部14类，不能少）----
    'classes': ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM',
                '128QAM', '256QAM', 'MSK'],

    'seed': 123,   # ★ 与原始不同的seed，保证数据多样性
}

SIG_LEN = CONFIG['signal_length']
SPS     = CONFIG['sps']
NUM_SYM = SIG_LEN // SPS
NUM_CLS = len(CONFIG['classes'])


# ══════════════════════════════════════════════
#  以下函数与原始数据生成代码完全相同
# ══════════════════════════════════════════════

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
    return (h / np.linalg.norm(h)).astype(np.float64)


_RRC = design_rrc(CONFIG['rrc_taps'], SPS, CONFIG['rolloff'])


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


def _make_psk(M):
    return np.exp(1j * 2 * np.pi * np.arange(M) / M)


def _make_square_qam(M):
    k = int(np.sqrt(M))
    assert k * k == M
    levels = 2.0 * np.arange(k) - (k - 1)
    gI, gQ = np.meshgrid(levels, levels)
    pts = (gI + 1j * gQ).flatten()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _make_cross_qam(M):
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
    c = CONSTELLATIONS[name]
    return c[np.random.randint(0, len(c), (n_batch, n_sym))]


def _gen_linear(const_name, n):
    syms = _rand_symbols(const_name, n)
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


def gen_pi4dqpsk(n):
    dphi = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    idx = np.random.randint(0, 4, (n, NUM_SYM))
    phase = np.cumsum(dphi[idx], axis=1)
    syms = np.exp(1j * phase)
    up = np.zeros((n, NUM_SYM * SPS), dtype=np.complex128)
    up[:, ::SPS] = syms
    return batch_fft_convolve(up, _RRC)[:, :SIG_LEN]


def gen_2ask(n):
    bits = np.random.randint(0, 2, (n, NUM_SYM)).astype(np.float64)
    up = np.zeros((n, NUM_SYM * SPS), dtype=np.complex128)
    up[:, ::SPS] = bits
    return batch_fft_convolve(up, _RRC)[:, :SIG_LEN]


def _gen_cpfsk(n, M, h_mod):
    syms = np.random.randint(0, M, (n, NUM_SYM))
    levels = 2.0 * np.arange(M) - (M - 1)
    freq_sym = levels[syms]
    freq_per_sample = h_mod * freq_sym / (2.0 * SPS)
    freq_up = np.repeat(freq_per_sample, SPS, axis=1)[:, :SIG_LEN]
    phase = 2 * np.pi * np.cumsum(freq_up, axis=1)
    return np.exp(1j * phase)


def gen_4fsk(n):
    M, h_mod = 4, 1.0 / 3.0
    syms = np.random.randint(0, M, (n, NUM_SYM))
    levels = 2.0 * np.arange(M) - (M - 1)
    freq_sym = levels[syms]
    freq_per_sample = h_mod * freq_sym / (2.0 * SPS)
    freq_up = np.repeat(freq_per_sample, SPS, axis=1)[:, :SIG_LEN]
    phase = 2 * np.pi * np.cumsum(freq_up, axis=1)
    return np.exp(1j * phase)


def gen_2fsk(n):  return _gen_cpfsk(n, M=2, h_mod=1.0)
def gen_msk(n):   return _gen_cpfsk(n, M=2, h_mod=0.5)


GENERATORS = {
    'BPSK': gen_bpsk,    'QPSK': gen_qpsk,      '8PSK': gen_8psk,
    'OQPSK': gen_oqpsk,  '2FSK': gen_2fsk,      '4FSK': gen_4fsk,
    '16QAM': gen_16qam,  '2ASK': gen_2ask,      'Pi4DQPSK': gen_pi4dqpsk,
    '64QAM': gen_64qam,  '32QAM': gen_32qam,    '128QAM': gen_128qam,
    '256QAM': gen_256qam, 'MSK': gen_msk,
}


class ChannelModel:
    def __init__(self, cfg=CONFIG):
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
        w = self.cfg['mixed_weights']
        types, probs = list(w.keys()), list(w.values())
        N = x.shape[0]
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
        return self._awgn(y, snr_db).astype(np.complex64)


class RFImpairments:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def apply_tx(self, x):
        if self.cfg['pa_enable']:
            p = self.cfg['pa_p']
            ibo = self.cfg['pa_ibo_db']
            pwr = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
            A_sat = np.sqrt(pwr * 10 ** (ibo / 10))
            r = np.abs(x)
            theta = np.angle(x)
            r_out = r / (1 + (r / (A_sat + 1e-12)) ** (2 * p)) ** (1 / (2 * p))
            x = r_out * np.exp(1j * theta)
        return x

    def apply_rx(self, x):
        if not self.cfg['impairments_enable']:
            return x
        # CFO
        N, L = x.shape
        df  = np.random.uniform(-self.cfg['cfo_max_norm'], self.cfg['cfo_max_norm'], (N, 1))
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        n   = np.arange(L).reshape(1, L)
        x   = x * np.exp(1j * (2 * np.pi * df * n + phi))
        # 相位噪声
        sigma = self.cfg['phase_noise_std']
        pn = np.cumsum(sigma * np.random.randn(N, L), axis=1)
        x  = x * np.exp(1j * pn)
        # IQ不平衡
        g_max = self.cfg['iq_gain_imb_max_db']
        p_max = self.cfg['iq_phase_imb_max_deg']
        eps  = 10 ** (np.random.uniform(-g_max, g_max, (N, 1)) / 20) - 1
        dphi = np.random.uniform(-p_max, p_max, (N, 1)) * np.pi / 180
        I, Q = x.real, x.imag
        I_out = (1 + eps / 2) * I
        Q_out = (1 - eps / 2) * (Q * np.cos(dphi) + I * np.sin(dphi))
        x = I_out + 1j * Q_out
        # DC偏移
        mx = self.cfg['dc_offset_max']
        dc = np.random.uniform(-mx, mx, (N, 1)) + 1j * np.random.uniform(-mx, mx, (N, 1))
        x  = x + dc
        # 时偏
        mx2 = self.cfg['timing_offset_max']
        offsets = np.random.uniform(-mx2, mx2, N)
        t = np.arange(L, dtype=np.float64)
        y = np.empty_like(x)
        for i in range(N):
            t_new = np.clip(t + offsets[i], 0, L - 1)
            y[i] = np.interp(t_new, t, x[i].real) + 1j * np.interp(t_new, t, x[i].imag)
        return y


def sample_gain_log_uniform(low, high, size):
    """
    ★ 核心修改：对数均匀采样
    原来 uniform(0.005, 0.5) 中 [0.005,0.015] 占比仅 ~2%
    改后 log_uniform(0.005, 0.5) 中 [0.005,0.015] 占比约 ~30%
    """
    log_low  = np.log10(low)
    log_high = np.log10(high)
    return 10 ** np.random.uniform(log_low, log_high, size)


def generate_all(cfg=CONFIG):
    np.random.seed(cfg['seed'])
    channel     = ChannelModel(cfg)
    impairments = RFImpairments(cfg)
    classes     = cfg['classes']
    snr_list    = cfg['snr_list_db']
    n_per       = cfg['n_per_class_per_snr']

    all_iq, all_y, all_snr = [], [], []
    total = len(classes) * len(snr_list)
    pbar  = tqdm(total=total, desc='Generating finetune data', ncols=90)

    for cls_idx, cls_name in enumerate(classes):
        gen_fn = GENERATORS[cls_name]
        for snr_db in snr_list:
            clean = gen_fn(n_per)
            clean = impairments.apply_tx(clean)
            noisy = channel.apply(clean, snr_db)
            noisy = impairments.apply_rx(noisy)

            # ★ 对数均匀增益采样（替代原来的 uniform）
            gain = sample_gain_log_uniform(
                cfg['gain_range'][0], cfg['gain_range'][1], size=(n_per, 1))

            noisy_scaled    = noisy * gain
            max_val         = 2 ** (cfg['adc_bit_depth'] - 1) - 1
            noisy_quantized = np.clip(noisy_scaled * max_val, -max_val - 1, max_val)
            noisy_quantized = np.round(noisy_quantized)
            noisy           = noisy_quantized / max_val

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
    print(f'\n总样本: {len(Y):,}   X shape: {X.shape}')
    return X, Y, SNR


def split_and_save(X, Y, SNR, cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    idx = np.arange(len(Y))
    i_tr, i_te = train_test_split(
        idx, test_size=1 - cfg['train_ratio'],
        random_state=cfg['seed'], stratify=Y)

    path = os.path.join(cfg['save_dir'], 'finetune_v2.npz')
    np.savez_compressed(
        path,
        X_train=X[i_tr], Y_train=Y[i_tr],
        X_test=X[i_te],  Y_test=Y[i_te],
        snr_test=SNR[i_te],
        classes=np.array(cfg['classes']),
        snr_list=np.array(cfg['snr_list_db'], dtype=np.float32))

    sz = os.path.getsize(path) / 1024 / 1024
    print(f'已保存 → {path}')
    print(f'  Train: {len(i_tr):,}  Test: {len(i_te):,}  Size: {sz:.1f} MB')
    return path


def verify_gain_distribution(npz_path):
    """验证增益分布是否符合预期"""
    data = np.load(npz_path, allow_pickle=True)
    X = data['X_train']
    # 用I路最大幅度近似估计增益水平
    max_amps = np.max(np.abs(X[:, 0, :]), axis=1)
    print('\n===== 增益分布验证 =====')
    print(f'  幅度 < 0.02  (低增益，对应实测场景): {np.mean(max_amps < 0.02)*100:.1f}%')
    print(f'  幅度 0.02~0.1 (中增益):               {np.mean((max_amps>=0.02)&(max_amps<0.1))*100:.1f}%')
    print(f'  幅度 > 0.1   (高增益):                {np.mean(max_amps > 0.1)*100:.1f}%')
    print('  预期：三个区间各约 25~40%')
    print('  若低增益占比仍<5%，请检查sample_gain_log_uniform是否生效')


if __name__ == '__main__':
    n_total = NUM_CLS * len(CONFIG['snr_list_db']) * CONFIG['n_per_class_per_snr']
    print('=' * 60)
    print('  Finetune Dataset Generator')
    print(f'  Classes:      {NUM_CLS} (全部14类)')
    print(f'  SNR range:    {CONFIG["snr_list_db"][0]}~{CONFIG["snr_list_db"][-1]} dB '
          f'({len(CONFIG["snr_list_db"])} steps)')
    print(f'  Per cls/SNR:  {CONFIG["n_per_class_per_snr"]}')
    print(f'  Total:        {n_total:,}')
    print(f'  Gain:         log_uniform({CONFIG["gain_range"][0]}, {CONFIG["gain_range"][1]})')
    print('=' * 60)

    X, Y, SNR = generate_all()
    path = split_and_save(X, Y, SNR)
    verify_gain_distribution(path)
    print('\n✓ 数据集生成完成！接下来运行 step2_finetune_ewc.py')
