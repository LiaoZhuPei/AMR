import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# %%
# 参数列表
CONFIG = {
    # 信号参数
    'sps': 4,
    'signal_length': 4800,
    'rolloff': 0.35,
    'rrc_taps': 101,

    # 数据规模
    'n_per_class_per_snr': 200,
    'snr_list_db': list(range(-2, 38, 2)),
    'train_ratio': 0.8,
    'save_dir': './sionna_dataset',

    # 信道
    'channel_type': 'mixed',
    'mixed_weights': {
        'awgn': 0.30,
        'flat_rayleigh': 0.20,
        'flat_rician': 0.15,
        'multipath': 0.35
    },
    'rician_k_db': 10.0,
    'multipath_taps': 8,
    'multipath_tau_rms': 3.0,

    # 发射端损伤
    'pa_enable': True,
    'pa_ibo_db': 8.0,
    'pa_p': 3,  # Rapp 模型平滑因子

    # 接收端损伤
    'impairments_enable': True,
    'cfo_max_norm': 0.001,  # 最大归一化频偏 Δf/fs
    'phase_noise_std': 0.010,  # Wiener 过程 σ (rad/sample)
    'iq_gain_imb_max_db': 1.0,  # 最大 IQ 增益不平衡 (dB)
    'iq_phase_imb_max_deg': 5.0,  # 最大 IQ 相位不平衡 (°)
    'dc_offset_max': 0.020,  # 最大 DC 偏移 (相对值)
    'timing_offset_max': 0.4,  # 最大分数采样偏移 (样点)

    'classes': ['OQPSK','Pi4DQPSK'],

    'seed': 62, #42

}
SIG_LEN = CONFIG['signal_length']
SPS = CONFIG['sps']
NUM_SYM = SIG_LEN // SPS
NUM_CLS = len(CONFIG['classes'])
L_HEAD_F = 768  # 192*sps=192*4=768
HEAD_SYM = 192
L_ALL = 4800


# %%
# 2. RRC 脉冲成形滤波器
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


# %%
# 3. FFT 批量卷积（替代逐样本 np.convolve）
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


# %%
# 5.信道模型
class ChannelModel:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    # 功率归一化
    def _norm(self, x):
        pwr = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
        return x / np.sqrt(pwr + 1e-12)

    # 衰落类型
    # 平坦瑞利信道
    def _flat_rayleigh(self, x):
        N = x.shape[0]  # 获取输入信号的长度（或批次大小）
        h = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)  # 信道系数
        # 信号x经过平坦瑞利信道后输出为x * h
        return x * h

    # 平坦莱斯衰落信道
    def _flat_rician(self, x):
        N = x.shape[0]
        K = 10 ** (self.cfg['rician_k_db'] / 10)
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        h_los = np.sqrt(K / (K + 1)) * np.exp(1j * phi)
        h_nlos = np.sqrt(1 / (2 * (K + 1))) * (np.random.randn(N, 1) + 1j * np.random.randn(N, 1))
        return x * (h_los + h_nlos)

    def _multipath(self, x):
        """
        多径频率选择性衰落
        指数功率时延谱: PDP(k) = exp(-k / τ_rms)
        每条径独立 Rayleigh 衰落
        """
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

    # 噪声
    def _awgn(self, x, snr_db):
        sigma2 = 10.0 ** (-snr_db / 10.0)
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
        return x + noise

    # 主接口
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


# %%
# 6.RF损伤
class RFImpairments:
    """
    施加顺序（符合实际硬件链路）:
      TX: PA 非线性
      RX: CFO → 相位噪声 → IQ 不平衡 → DC 偏移 → 采样时偏
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

    # TX侧
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

    # RX侧
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
        df = np.random.uniform(-max_cfo, max_cfo, (N, 1))
        phi = np.random.uniform(0, 2 * np.pi, (N, 1))
        n = np.arange(L).reshape(1, L)
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
        # eps → ε
        eps = 10 ** (np.random.uniform(-g_max, g_max, (N, 1)) / 20) - 1  # dB → 线性偏差
        # dphi → Δφ
        dphi = np.random.uniform(-p_max, p_max, (N, 1)) * np.pi / 180

        I, Q = x.real, x.imag
        I_out = (1 + eps) * I
        Q_out = (1 - eps) * (Q * np.cos(dphi) + I * np.sin(dphi))
        return I_out + 1j * Q_out

    def _dc_offset(self, x):
        """ADC 直流泄漏"""
        N, L = x.shape
        mx = self.cfg['dc_offset_max']
        if mx <= 0:
            return x
        dc = np.random.uniform(-mx, mx, (N, 1)) + 1j * np.random.uniform(-mx, mx, (N, 1))
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
            y[i] = np.interp(t_new, t, x[i].real) + 1j * np.interp(t_new, t, x[i].imag)
        return y


# %%
# B. OQPSK: Q 路半符号延迟
def zc_seq(num_sym, u=25):
    """
    生成Zadoff-Chu序列
    num_sym: 符号长度(192)
    u:       根序列索引
    返回shape: (num_sym,) 复数
    """
    n = np.arange(num_sym)
    zc = np.exp(-1j * np.pi * u * n * (n + 1) / num_sym)
    return zc


def gen_oqpsk(n, NUM_SYM):
    """
    n:       batch size
    NUM_SYM: 符号数
    返回shape: (n, NUM_SYM*SPS)
    """
    bi = 2.0 * np.random.randint(0, 2, (n, NUM_SYM)) - 1.0
    bq = 2.0 * np.random.randint(0, 2, (n, NUM_SYM)) - 1.0
    L_data = NUM_SYM * SPS  # 4800-768

    up_i = np.zeros((n, L_data))
    up_i[:, ::SPS] = bi

    up_q = np.zeros((n, L_data))
    q_idx = np.arange(NUM_SYM) * SPS + SPS // 2
    q_idx = q_idx[q_idx < L_data]
    up_q[:, q_idx] = bq[:, :len(q_idx)]

    # si = batch_fft_convolve(up_i, _RRC).real
    # sq = batch_fft_convolve(up_q, _RRC).real

    sig_data = (up_i + 1j * up_q)[:, :L_data]

    return sig_data


def gen_zc_head(n):
    """
    生成经过RRC滤波并截取的ZC同步头
    返回shape: (n, L_HEAD_F=768)
    """
    zc = zc_seq(HEAD_SYM)  # (192,)

    # sps
    zc_up = np.zeros(HEAD_SYM * SPS, dtype=complex)  # (768, )
    zc_up[::SPS] = zc

    zc_up = np.tile(zc_up, (n, 1))

    # zc_filtered = batch_fft_convolve(zc_up, _RRC)[:, :L_HEAD_F]
    return zc_up


def gen_head_oqpsk(n):
    """
    随机生成Mode A或Mode B的混合信号
    返回shape: (n, L_ALL=4800)
    """
    #mode = np.random.randint(0, 1)
    mode = 1
    sig_head = gen_zc_head(n)

    if mode == 0:
        # ── Mode A：同步头在头部 ──
        # OQPSK符号数 = (4800 - 768) / 4 = 1008
        NUM_SYM_DATA = (L_ALL - L_HEAD_F) // SPS  # 1008
        sig_data = gen_oqpsk(n, NUM_SYM_DATA)  # shape(n, 4032)
        sig_all = np.concatenate([sig_head, sig_data], axis=1)
    else:
        # ── Mode B：同步头在中间 ──
        # 从[1400, 1600, 1800, 2000]中随机选一个采样点位置
        candiates = np.arange(1400, 2200, 200)  # [1400,1600,1800,2000]
        head_mid_position = int(np.random.choice(candiates))

        # data1符号数
        NUM_SYM_1 = head_mid_position // SPS
        # data2符号数
        NUM_SYM_2 = (L_ALL - L_HEAD_F - head_mid_position) // SPS

        sig_data1 = gen_oqpsk(n, NUM_SYM_1)
        sig_data2 = gen_oqpsk(n, NUM_SYM_2)

        sig_all = np.concatenate(
            [sig_data1, sig_head, sig_data2], axis=1
        )
    return batch_fft_convolve(sig_all, _RRC)[:, :L_ALL]


# %%
# C. π/4-DQPSK: 差分编码 + π/4 旋转
def gen_pi4dqpsk(n, NUM_SYM):
    dphi = np.array([np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
    idx = np.random.randint(0, 4, (n, NUM_SYM))
    phase = np.cumsum(dphi[idx], axis=1)  # 矢量化差分累积
    syms = np.exp(1j * phase)

    # up = np.zeros((n, NUM_SYM * SPS), dtype=np.complex128)
    # up[:, ::SPS] = syms
    return syms


def gen_head_pi4dqpsk(n):
    """
    随机生成Mode A或Mode B的混合信号
    返回shape: (n, L_ALL=4800)
    """
    # mode = np.random.randint(0, 1)
    mode = 1
    sig_head_sym = np.tile(zc_seq(HEAD_SYM), (n, 1))

    if mode == 0:
        # ── Mode A：同步头在头部 ──
        # OQPSK符号数 = (4800 - 768) / 4 = 1008
        NUM_SYM_DATA = (L_ALL - L_HEAD_F) // SPS  # 1008
        sig_data_sym = gen_pi4dqpsk(n, NUM_SYM_DATA)  # shape(n, 4032)
        sig_all_sym = np.concatenate([sig_head_sym, sig_data_sym], axis=1)
    else:
        # ── Mode B：同步头在中间 ──
        # 从[1400, 1600, 1800, 2000]中随机选一个采样点位置
        candiates = np.arange(1400, 2200, 200)  # [1400,1600,1800,2000]
        head_mid_position = int(np.random.choice(candiates))

        # data1符号数
        NUM_SYM_1 = head_mid_position // SPS
        # data2符号数
        NUM_SYM_2 = (L_ALL - L_HEAD_F - head_mid_position) // SPS

        sig_data1_sym = gen_pi4dqpsk(n, NUM_SYM_1)
        sig_data2_sym = gen_pi4dqpsk(n, NUM_SYM_2)

        sig_all_sym = np.concatenate(
            [sig_data1_sym, sig_head_sym, sig_data2_sym], axis=1
        )
    sig_all = np.zeros((n, L_ALL), dtype=np.complex128)
    sig_all[:, ::SPS] = sig_all_sym
    return batch_fft_convolve(sig_all, _RRC)[:, :L_ALL]


# %%
# 8.生成器注册表
GENERATORS = {
    'OQPSK': gen_head_oqpsk, 'Pi4DQPSK': gen_head_pi4dqpsk,
}


# %%
# 9.主生成
def generate_all(cfg=CONFIG):
    np.random.seed(cfg['seed'])

    channel = ChannelModel(cfg)
    impairments = RFImpairments(cfg)

    classes = cfg['classes']
    snr_list = cfg['snr_list_db']
    n_per = cfg['n_per_class_per_snr']

    all_iq, all_y, all_snr = [], [], []
    total = len(classes) * len(snr_list)
    pbar = tqdm(total=total, desc='Generating', ncols=90)

    for cls_idx, cls_name in enumerate(classes):
        gen_fn = GENERATORS[cls_name]

        for snr_db in snr_list:
            # ① 生成干净基带信号
            clean = gen_fn(n_per)  # (N, SIG_LEN) complex

            # ② TX 损伤 (PA)
            clean = impairments.apply_tx(clean)

            # ③ 信道 (衰落 + AWGN)
            noisy = channel.apply(clean, snr_db)

            # ④ RX 损伤 (CFO, 相噪, IQ不平衡, DC, 时偏)
            noisy = impairments.apply_rx(noisy)

            # ⑤ 转 IQ: (N, 2, L)
            iq = np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)

            all_iq.append(iq)
            all_y.extend([cls_idx] * n_per)
            all_snr.extend([float(snr_db)] * n_per)

            pbar.set_postfix(cls=cls_name, snr=f'{snr_db:+d}dB')
            pbar.update(1)

    pbar.close()

    X = np.concatenate(all_iq, axis=0)
    Y = np.array(all_y, dtype=np.int64)
    SNR = np.array(all_snr, dtype=np.float32)

    print(f'\n 总样本：{len(Y):,}  X: {X.shape}  '
          f'SNR: {SNR.min():.0f}~{SNR.max():.0f} dB')
    return X, Y, SNR


# %%
# 10. 保存
def split_and_save(X, Y, SNR, cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    idx = np.arange(len(Y))
    i_tr, i_te = train_test_split(idx, test_size=1 - cfg['train_ratio'], random_state=cfg['seed'], stratify=Y)

    path = os.path.join(cfg['save_dir'], 'Sionna_head_mid1.npz')
    np.savez_compressed(path,
                        X_train=X[i_tr], Y_train=Y[i_tr],
                        X_test=X[i_te], Y_test=Y[i_te],
                        snr_test=SNR[i_te],
                        classes=np.array(cfg['classes']),
                        snr_list=np.array(cfg['snr_list_db'], dtype=np.float32))

    sz = os.path.getsize(path) / 1024 / 1024
    print(f' 已保存 → {path}')
    print(f' Train:{len(i_tr):,} Test: {len(i_te):,} Size:{sz:.1f} MB')
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