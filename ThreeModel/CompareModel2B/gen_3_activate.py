import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION — 动态范围配置                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
CONFIG = {
    # ---- 信号参数 (固定与动态) ----
    'signal_length': 1024,  # 固定长度
    'sps_choices': [4, 4],  # 动态范围: 以4为中心
    'rolloff_range': (0.30, 0.40),  # 动态范围: 以0.35为中心
    'rrc_taps': 101,  # 固定滤波器长度

    # ---- 数据集规模 (固定不变) ----
    'n_per_class_per_snr': 3200,
    'chunk_size': 320,  # 分批生成以引入动态多样性 (3200/320=10批)
    'snr_list_db': list(range(-2, 38, 2)),  # 20 档
    'train_ratio': 0.8,
    'save_dir': './sionna_dynamic_dataset',

    # ---- 信道 (全动态范围) ----
    'channel_type': 'mixed',
    'mixed_weights': {
        'awgn': 0.20, 'flat_rayleigh': 0.20, 'flat_rician': 0.20, 'multipath': 0.40
    },
    'rician_k_db_range': (5.0, 15.0),  # Rician K因子范围
    'multipath_taps_range': (4, 12),  # 多径抽头数范围
    'multipath_tau_rms_range': (1.0, 5.0),  # 时延扩展RMS范围

    # ---- 发射端损伤 (全动态范围) ----
    'pa_enable': True,
    'pa_ibo_db_range': (4.0, 12.0),  # 功率放大器输入回退范围
    'pa_p_range': (2.0, 4.0),  # Rapp模型平滑因子范围

    # ---- 接收端损伤 (全动态范围) ----
    'impairments_enable': True,
    'cfo_max_norm_range': (0.005, 0.02),  # 最大归一化频偏范围 Δf/fs
    'phase_noise_std_range': (0.005, 0.02),  # 相位噪声标准差范围
    'iq_gain_imb_db_range': (0.5, 3.5),  # IQ增益不平衡(dB)范围
    'iq_phase_imb_deg_range': (3.0, 15.0),  # IQ相位不平衡(°)范围
    'dc_offset_max_range': (0.005, 0.03),  # DC偏移范围
    'timing_offset_range': (0.1, 0.7),  # 最大分数采样偏移范围

    # ---- 调制类别 ----
    'classes': ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
                '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM',
                '128QAM', '256QAM', 'MSK'],

    # ---- 可复现性 ----
    'seed': 48,
}

SIG_LEN = CONFIG['signal_length']
NUM_CLS = len(CONFIG['classes'])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2. 核心数学与滤波器工具                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
def design_rrc(ntaps, sps, beta):
    """根据动态的 sps 和 rolloff(beta) 实时设计 RRC 滤波器"""
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


def batch_fft_convolve(signals, filt):
    """矢量化 FFT 卷积"""
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
# ║  3. 星座定义 (维持原样)                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _make_psk(M): return np.exp(1j * 2 * np.pi * np.arange(M) / M)


def _make_square_qam(M):
    k = int(np.sqrt(M))
    levels = 2.0 * np.arange(k) - (k - 1)
    gI, gQ = np.meshgrid(levels, levels)
    pts = (gI + 1j * gQ).flatten()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _make_cross_qam(M):
    if M == 32:
        levels = np.array([-5, -3, -1, 1, 3, 5], dtype=np.float64)
        gI, gQ = np.meshgrid(levels, levels)
        pts = (gI + 1j * gQ).flatten()
        mask = ~((np.abs(pts.real) == 5) & (np.abs(pts.imag) == 5))
        pts = pts[mask]
    elif M == 128:
        levels = np.arange(-11, 12, 2, dtype=np.float64)
        gI, gQ = np.meshgrid(levels, levels)
        pts = (gI + 1j * gQ).flatten()
        mask = ~((np.abs(pts.real) >= 9) & (np.abs(pts.imag) >= 9))
        pts = pts[mask]
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS = {
    'BPSK': _make_psk(2), 'QPSK': _make_psk(4), '8PSK': _make_psk(8),
    '16QAM': _make_square_qam(16), '32QAM': _make_cross_qam(32),
    '64QAM': _make_square_qam(64), '128QAM': _make_cross_qam(128),
    '256QAM': _make_square_qam(256),
}


def _rand_symbols(name, n_batch, n_sym):
    c = CONSTELLATIONS[name]
    return c[np.random.randint(0, len(c), (n_batch, n_sym))]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4. 动态信道模型                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
class ChannelModel:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def _norm(self, x):
        return x / np.sqrt(np.mean(np.abs(x) ** 2, axis=-1, keepdims=True) + 1e-12)

    def _flat_rayleigh(self, x):
        N = x.shape[0]
        h = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)
        return x * h

    def _flat_rician(self, x):
        N = x.shape[0]
        # 动态抽取 K 因子 (为每个样本独立分配一个 K)
        k_db = np.random.uniform(*self.cfg['rician_k_db_range'], (N, 1))
        K = 10 ** (k_db / 10)
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        h_los = np.sqrt(K / (K + 1)) * np.exp(1j * phi)
        h_nlos = np.sqrt(1 / (2 * (K + 1))) * (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
        return x * (h_los + h_nlos)

    def _multipath(self, x):
        N, L = x.shape
        # 动态多径参数
        n_taps = np.random.randint(self.cfg['multipath_taps_range'][0], self.cfg['multipath_taps_range'][1] + 1)
        tau_rms = np.random.uniform(*self.cfg['multipath_tau_rms_range'], (N, 1))

        # 矢量化计算多径PDP
        k = np.arange(n_taps, dtype=np.float64)[np.newaxis, :]
        pdp = np.exp(-k / tau_rms)
        pdp /= pdp.sum(axis=1, keepdims=True)

        h = np.sqrt(pdp / 2) * (np.random.randn(N, n_taps) + 1j * np.random.randn(N, n_taps))

        y = np.zeros_like(x)
        for tap in range(n_taps):
            if tap == 0:
                y += h[:, 0:1] * x
            else:
                y[:, tap:] += h[:, tap:tap + 1] * x[:, :-tap]
        return y

    def apply(self, signals, snr_db):
        x = self._norm(signals)
        ch = self.cfg['channel_type']

        if ch == 'mixed':
            w = self.cfg['mixed_weights']
            types, probs = list(w.keys()), list(w.values())
            assign = np.random.choice(types, size=x.shape[0], p=probs)
            y = x.copy()
            for ct in types:
                m = (assign == ct)
                if not m.any(): continue
                sub = x[m]
                if ct == 'flat_rayleigh':
                    sub = self._flat_rayleigh(sub)
                elif ct == 'flat_rician':
                    sub = self._flat_rician(sub)
                elif ct == 'multipath':
                    sub = self._multipath(sub)
                y[m] = sub
            x = y
        elif ch == 'flat_rayleigh':
            x = self._flat_rayleigh(x)
        elif ch == 'flat_rician':
            x = self._flat_rician(x)
        elif ch == 'multipath':
            x = self._multipath(x)

        sigma2 = 10.0 ** (-snr_db / 10.0)
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
        return (x + noise).astype(np.complex64)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5. 动态 RF 损伤                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
class RFImpairments:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    def apply_tx(self, x):
        if not self.cfg['pa_enable']: return x
        N = x.shape[0]
        # 动态提取 PA 参数
        ibo = np.random.uniform(*self.cfg['pa_ibo_db_range'], (N, 1))
        p = np.random.uniform(*self.cfg['pa_p_range'], (N, 1))

        pwr = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
        A_sat = np.sqrt(pwr * 10 ** (ibo / 10))
        r, theta = np.abs(x), np.angle(x)
        r_out = r / (1 + (r / (A_sat + 1e-12)) ** (2 * p)) ** (1 / (2 * p))
        return r_out * np.exp(1j * theta)

    def apply_rx(self, x):
        if not self.cfg['impairments_enable']: return x
        N, L = x.shape

        # 1. 动态载波频偏 CFO
        cfo_limit = np.random.uniform(*self.cfg['cfo_max_norm_range'], (N, 1))
        df = np.random.uniform(-cfo_limit, cfo_limit)
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        x = x * np.exp(1j * (2 * np.pi * df * np.arange(L).reshape(1, L) + phi))

        # 2. 动态相位噪声
        sigma = np.random.uniform(*self.cfg['phase_noise_std_range'], (N, 1))
        pn = np.cumsum(sigma * np.random.randn(N, L), axis=1)
        x = x * np.exp(1j * pn)

        # 3. 动态 IQ 不平衡
        g_limit = np.random.uniform(*self.cfg['iq_gain_imb_db_range'], (N, 1))
        p_limit = np.random.uniform(*self.cfg['iq_phase_imb_deg_range'], (N, 1))
        eps = 10 ** (np.random.uniform(-g_limit, g_limit) / 20) - 1
        dphi = np.random.uniform(-p_limit, p_limit) * np.pi / 180
        I, Q = x.real, x.imag
        x = ((1 + eps / 2) * I) + 1j * ((1 - eps / 2) * (Q * np.cos(dphi) + I * np.sin(dphi)))

        # 4. 动态 DC 偏移
        dc_limit = np.random.uniform(*self.cfg['dc_offset_max_range'], (N, 1))
        x += np.random.uniform(-dc_limit, dc_limit) + 1j * np.random.uniform(-dc_limit, dc_limit)

        # 5. 动态时偏 (分数采样插值)
        t_limit = np.random.uniform(*self.cfg['timing_offset_range'], N)
        offsets = np.random.uniform(-t_limit, t_limit)
        t = np.arange(L, dtype=np.float64)
        y = np.empty_like(x)
        for i in range(N):
            t_new = np.clip(t + offsets[i], 0, L - 1)
            y[i] = np.interp(t_new, t, x[i].real) + 1j * np.interp(t_new, t, x[i].imag)

        return y


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6. 调制信号生成器（适配动态 SPS 与 Rolloff）                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _gen_linear(const_name, n, sps, rolloff):
    num_sym = int(np.ceil(SIG_LEN / sps)) + 4  # 加入余量防截断
    syms = _rand_symbols(const_name, n, num_sym)
    up = np.zeros((n, num_sym * sps), dtype=np.complex128)
    up[:, ::sps] = syms
    rrc = design_rrc(CONFIG['rrc_taps'], sps, rolloff)
    return batch_fft_convolve(up, rrc)[:, :SIG_LEN]


def gen_oqpsk(n, sps, rolloff):
    num_sym = int(np.ceil(SIG_LEN / sps)) + 4
    bi = 2.0 * np.random.randint(0, 2, (n, num_sym)) - 1.0
    bq = 2.0 * np.random.randint(0, 2, (n, num_sym)) - 1.0
    L = num_sym * sps
    up_i, up_q = np.zeros((n, L)), np.zeros((n, L))
    up_i[:, ::sps] = bi

    q_idx = np.arange(num_sym) * sps + sps // 2
    q_idx = q_idx[q_idx < L]
    up_q[:, q_idx] = bq[:, :len(q_idx)]

    rrc = design_rrc(CONFIG['rrc_taps'], sps, rolloff)
    si = batch_fft_convolve(up_i, rrc).real
    sq = batch_fft_convolve(up_q, rrc).real
    return (si + 1j * sq)[:, :SIG_LEN]


def gen_pi4dqpsk(n, sps, rolloff):
    num_sym = int(np.ceil(SIG_LEN / sps)) + 4
    dphi = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    phase = np.cumsum(dphi[np.random.randint(0, 4, (n, num_sym))], axis=1)
    up = np.zeros((n, num_sym * sps), dtype=np.complex128)
    up[:, ::sps] = np.exp(1j * phase)
    rrc = design_rrc(CONFIG['rrc_taps'], sps, rolloff)
    return batch_fft_convolve(up, rrc)[:, :SIG_LEN]


def gen_2ask(n, sps, rolloff):
    num_sym = int(np.ceil(SIG_LEN / sps)) + 4
    bits = np.random.randint(0, 2, (n, num_sym)).astype(np.float64)
    up = np.zeros((n, num_sym * sps), dtype=np.complex128)
    up[:, ::sps] = bits
    rrc = design_rrc(CONFIG['rrc_taps'], sps, rolloff)
    return batch_fft_convolve(up, rrc)[:, :SIG_LEN]


def _gen_cpfsk(n, M, h_mod, sps):
    num_sym = int(np.ceil(SIG_LEN / sps)) + 4
    freq_sym = 2.0 * np.random.randint(0, M, (n, num_sym)) - (M - 1)
    freq_per_sample = h_mod * freq_sym / (2.0 * sps)
    freq_up = np.repeat(freq_per_sample, sps, axis=1)[:, :SIG_LEN]
    return np.exp(1j * 2 * np.pi * np.cumsum(freq_up, axis=1))


# 统一包装函数
def wrap_lin(name): return lambda n, s, r: _gen_linear(name, n, s, r)


GENERATORS = {
    'BPSK': wrap_lin('BPSK'), 'QPSK': wrap_lin('QPSK'), '8PSK': wrap_lin('8PSK'),
    '16QAM': wrap_lin('16QAM'), '32QAM': wrap_lin('32QAM'), '64QAM': wrap_lin('64QAM'),
    '128QAM': wrap_lin('128QAM'), '256QAM': wrap_lin('256QAM'),
    'OQPSK': gen_oqpsk, 'Pi4DQPSK': gen_pi4dqpsk, '2ASK': gen_2ask,
    '2FSK': lambda n, s, r: _gen_cpfsk(n, 2, 1.0, s),
    '4FSK': lambda n, s, r: _gen_cpfsk(n, 4, 1 / 3, s),
    'MSK': lambda n, s, r: _gen_cpfsk(n, 2, 0.5, s),
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7. 主流水线：引入 Chunk 以实现样本多样性                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
def generate_all(cfg=CONFIG):
    np.random.seed(cfg['seed'])
    channel = ChannelModel(cfg)
    impairments = RFImpairments(cfg)

    classes, snr_list = cfg['classes'], cfg['snr_list_db']
    n_per, chunk_size = cfg['n_per_class_per_snr'], cfg['chunk_size']
    n_chunks = n_per // chunk_size

    all_iq, all_y, all_snr = [], [], []
    total = len(classes) * len(snr_list)
    pbar = tqdm(total=total, desc='Generating Dynamic Dataset', ncols=100)

    for cls_idx, cls_name in enumerate(classes):
        gen_fn = GENERATORS[cls_name]

        for snr_db in snr_list:
            for _ in range(n_chunks):
                # ★ 为当前 Chunk 动态抽取 SPS 和 Rolloff ★
                sps = np.random.choice(cfg['sps_choices'])
                rolloff = np.random.uniform(*cfg['rolloff_range'])

                # ① 生成干净基带信号
                clean = gen_fn(chunk_size, sps, rolloff)
                # ② TX 损伤
                clean = impairments.apply_tx(clean)
                # ③ 信道
                noisy = channel.apply(clean, snr_db)
                # ④ RX 损伤
                noisy = impairments.apply_rx(noisy)
                # ⑤ 转 IQ
                iq = np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)

                all_iq.append(iq)
                all_y.extend([cls_idx] * chunk_size)
                all_snr.extend([float(snr_db)] * chunk_size)

            pbar.set_postfix(cls=cls_name, snr=f'{snr_db:+d}dB')
            pbar.update(1)

    pbar.close()
    X = np.concatenate(all_iq, axis=0)
    Y = np.array(all_y, dtype=np.int64)
    SNR = np.array(all_snr, dtype=np.float32)
    return X, Y, SNR


def split_and_save(X, Y, SNR, cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    idx = np.arange(len(Y))
    i_tr, i_te = train_test_split(idx, test_size=1 - cfg['train_ratio'], random_state=cfg['seed'], stratify=Y)

    path = os.path.join(cfg['save_dir'], 'Activate_AMR_Dataset.npz')
    np.savez_compressed(path, X_train=X[i_tr], Y_train=Y[i_tr], X_test=X[i_te], Y_test=Y[i_te],
                        snr_test=SNR[i_te], classes=np.array(cfg['classes']), snr_list=np.array(cfg['snr_list_db']))
    print(f'\n  Saved → {path}  Size: {os.path.getsize(path) / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    print('=' * 65)
    print('  AMR Dataset Generator (Fully Dynamic Configuration)')
    print('=' * 65)
    X, Y, SNR = generate_all()
    split_and_save(X, Y, SNR)