"""
单调制类型仿真数据生成器
用法: python gen_single_mod.py --mod 128QAM --snr 20 --n 2000 --out ./sim_128QAM.npz
"""

import os
import argparse
import numpy as np

# ── 参数 ──────────────────────────────────────────────────────────────
CONFIG = {
    'signal_length':      1024,
    'sps':                4,
    'rolloff':            0.35,
    'rrc_taps':           101,
    'pa_enable':          True,
    'pa_ibo_db':          8.0,
    'pa_p':               3,
    'impairments_enable': True,
    'cfo_max_norm':       0.005,
    'phase_noise_std':    0.015,
    'iq_gain_imb_max_db': 1.5,
    'iq_phase_imb_max_deg': 5.0,
    'dc_offset_max':      0.020,
    'timing_offset_max':  0.4,
    'adc_bit_depth':      16,
    'gain_range':         [0.005, 0.5],
    'channel_type':       'mixed',
    'mixed_weights':      {'awgn': 0.30, 'flat_rayleigh': 0.20,
                           'flat_rician': 0.15, 'multipath': 0.35},
    'rician_k_db':        10.0,
    'multipath_taps':     8,
    'multipath_tau_rms':  3.0,
    'seed':               45,
}

SIG_LEN = CONFIG['signal_length']
SPS     = CONFIG['sps']
NUM_SYM = SIG_LEN // SPS


# ── RRC ───────────────────────────────────────────────────────────────
def design_rrc(ntaps, sps, beta):
    T = float(sps)
    n = np.arange(-(ntaps // 2), ntaps // 2 + 1, dtype=np.float64)
    h = np.zeros_like(n)
    for i, t in enumerate(n):
        if abs(t) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(t) - T / (4.0 * beta)) < 1e-12:
            h[i] = beta / np.sqrt(2) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
        else:
            num = (np.sin(np.pi*t/T*(1-beta)) +
                   4*beta*t/T*np.cos(np.pi*t/T*(1+beta)))
            den = np.pi*t/T*(1-(4*beta*t/T)**2)
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


# ── 星座 ──────────────────────────────────────────────────────────────
def _make_square_qam(M):
    k = int(np.sqrt(M))
    levels = 2.0 * np.arange(k) - (k - 1)
    gI, gQ = np.meshgrid(levels, levels)
    pts = (gI + 1j * gQ).flatten()
    return pts / np.sqrt(np.mean(np.abs(pts)**2))

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
        idx = np.argsort(np.abs(pts)**2)[:128]
        pts = pts[idx]
    else:
        raise ValueError(f"Cross-QAM 未定义 M={M}")
    return pts / np.sqrt(np.mean(np.abs(pts)**2))

def _make_psk(M):
    return np.exp(1j * 2 * np.pi * np.arange(M) / M)

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

def _gen_linear(name, n):
    c = CONSTELLATIONS[name]
    syms = c[np.random.randint(0, len(c), (n, NUM_SYM))]
    up = np.zeros((n, NUM_SYM * SPS), dtype=np.complex128)
    up[:, ::SPS] = syms
    return batch_fft_convolve(up, _RRC)[:, :SIG_LEN]

def gen_128qam(n): return _gen_linear('128QAM', n)
def gen_64qam(n):  return _gen_linear('64QAM',  n)
def gen_32qam(n):  return _gen_linear('32QAM',  n)
def gen_16qam(n):  return _gen_linear('16QAM',  n)
def gen_256qam(n): return _gen_linear('256QAM', n)

GENERATORS = {
    '16QAM': gen_16qam, '32QAM': gen_32qam,
    '64QAM': gen_64qam, '128QAM': gen_128qam, '256QAM': gen_256qam,
}


# ── 信道 ──────────────────────────────────────────────────────────────
def _norm(x):
    pwr = np.mean(np.abs(x)**2, axis=-1, keepdims=True)
    return x / np.sqrt(pwr + 1e-12)

def _flat_rayleigh(x):
    N = x.shape[0]
    h = (np.random.randn(N,1) + 1j*np.random.randn(N,1)) / np.sqrt(2)
    return x * h

def _flat_rician(x):
    N = x.shape[0]
    K = 10**(CONFIG['rician_k_db']/10)
    phi = np.random.uniform(0, 2*np.pi, (N,1))
    h_los  = np.sqrt(K/(K+1)) * np.exp(1j*phi)
    h_nlos = np.sqrt(1/(2*(K+1))) * (np.random.randn(N,1)+1j*np.random.randn(N,1))
    return x * (h_los + h_nlos)

def _multipath(x):
    N, L = x.shape
    n_taps = CONFIG['multipath_taps']
    tau    = CONFIG['multipath_tau_rms']
    pdp = np.exp(-np.arange(n_taps)/tau); pdp /= pdp.sum()
    h = np.zeros((N, n_taps), dtype=np.complex128)
    for k in range(n_taps):
        h[:,k] = np.sqrt(pdp[k]/2)*(np.random.randn(N)+1j*np.random.randn(N))
    y = np.zeros_like(x)
    for k in range(n_taps):
        if k == 0: y += h[:,0:1]*x
        else:      y[:,k:] += h[:,k:k+1]*x[:,:-k]
    return y

def apply_channel(x, snr_db):
    x = _norm(x)
    w = CONFIG['mixed_weights']
    types, probs = list(w.keys()), list(w.values())
    assign = np.random.choice(types, size=x.shape[0], p=probs)
    y = x.copy()
    for ch in types:
        m = (assign == ch)
        if not m.any(): continue
        sub = x[m]
        if ch == 'flat_rayleigh': sub = _flat_rayleigh(sub)
        elif ch == 'flat_rician': sub = _flat_rician(sub)
        elif ch == 'multipath':   sub = _multipath(sub)
        y[m] = sub
    sigma2 = 10.0**(-snr_db/10.0)
    noise = np.sqrt(sigma2/2)*(np.random.randn(*y.shape)+1j*np.random.randn(*y.shape))
    return (y + noise).astype(np.complex64)


# ── RF 损伤 ────────────────────────────────────────────────────────────
def apply_pa(x):
    p, ibo = CONFIG['pa_p'], CONFIG['pa_ibo_db']
    pwr = np.mean(np.abs(x)**2, axis=-1, keepdims=True)
    A_sat = np.sqrt(pwr * 10**(ibo/10))
    r = np.abs(x); theta = np.angle(x)
    r_out = r / (1+(r/(A_sat+1e-12))**(2*p))**(1/(2*p))
    return r_out * np.exp(1j*theta)

def apply_rx(x):
    N, L = x.shape
    # CFO
    df  = np.random.uniform(-CONFIG['cfo_max_norm'], CONFIG['cfo_max_norm'], (N,1))
    phi = np.random.uniform(0, 2*np.pi, (N,1))
    n   = np.arange(L).reshape(1,L)
    x   = x * np.exp(1j*(2*np.pi*df*n+phi))
    # 相位噪声
    pn = np.cumsum(CONFIG['phase_noise_std']*np.random.randn(N,L), axis=1)
    x  = x * np.exp(1j*pn)
    # IQ 不平衡
    eps  = 10**(np.random.uniform(-CONFIG['iq_gain_imb_max_db'], CONFIG['iq_gain_imb_max_db'], (N,1))/20)-1
    dphi = np.random.uniform(-CONFIG['iq_phase_imb_max_deg'], CONFIG['iq_phase_imb_max_deg'], (N,1))*np.pi/180
    I, Q = x.real, x.imag
    x = (1+eps/2)*I + 1j*(1-eps/2)*(Q*np.cos(dphi)+I*np.sin(dphi))
    # DC 偏移
    dc = (np.random.uniform(-CONFIG['dc_offset_max'], CONFIG['dc_offset_max'], (N,1)) +
          1j*np.random.uniform(-CONFIG['dc_offset_max'], CONFIG['dc_offset_max'], (N,1)))
    x  = x + dc
    # 时间偏移
    offsets = np.random.uniform(-CONFIG['timing_offset_max'], CONFIG['timing_offset_max'], N)
    t = np.arange(L, dtype=np.float64)
    y = np.empty_like(x)
    for i in range(N):
        t_new = np.clip(t+offsets[i], 0, L-1)
        y[i]  = np.interp(t_new, t, x[i].real)+1j*np.interp(t_new, t, x[i].imag)
    return y

def apply_adc(x):
    gain = np.random.uniform(CONFIG['gain_range'][0], CONFIG['gain_range'][1], (x.shape[0],1))
    max_val = 2**(CONFIG['adc_bit_depth']-1)-1
    q = np.round(np.clip(x*gain*max_val, -max_val-1, max_val))
    return q / max_val


# ── 主生成函数 ─────────────────────────────────────────────────────────
def generate(mod: str, n: int, snr_db: float) -> np.ndarray:
    """
    生成 n 个指定调制在指定 SNR 下的 IQ 片段
    返回: (n, 2, 1024) float32，[0]=I，[1]=Q
    """
    if mod not in GENERATORS:
        raise ValueError(f"不支持的调制类型: {mod}，可选: {list(GENERATORS.keys())}")
    gen_fn = GENERATORS[mod]
    clean  = gen_fn(n)
    clean  = apply_pa(clean)
    noisy  = apply_channel(clean, snr_db)
    noisy  = apply_rx(noisy)
    noisy  = apply_adc(noisy)
    return np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='单调制类型仿真数据生成器')
    parser.add_argument('--mod', default='128QAM',
                        choices=list(GENERATORS.keys()), help='调制类型')
    parser.add_argument('--snr', type=float, default=20.0, help='SNR (dB)')
    parser.add_argument('--n',   type=int,   default=2000, help='生成样本数')
    parser.add_argument('--out', default='./sim_single.npz', help='输出 .npz 路径')
    parser.add_argument('--seed', type=int,  default=45)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print(f'生成 {args.mod}  SNR={args.snr:+.0f}dB  n={args.n} ...')
    X = generate(args.mod, args.n, args.snr)
    np.savez_compressed(args.out, X=X, mod=args.mod, snr=args.snr)
    sz = os.path.getsize(args.out)/1024/1024
    print(f'✓ 保存至 {args.out}  ({sz:.1f} MB)  形状: {X.shape}')
