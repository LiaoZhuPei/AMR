#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_sionna_data.py  —— 纯 NumPy 版本（无需 Sionna / TensorFlow）

生成 AMR（自动调制识别）仿真数据集
  ● 14 种调制：BPSK, QPSK, 8PSK, OQPSK, 2FSK, 4FSK, 16QAM, 2ASK,
               Pi4DQPSK, 64QAM, 32QAM, 128QAM, 256QAM, MSK
  ● RRC 脉冲成形
  ● AWGN / Rayleigh / Rician 信道
  ● 可选射频损伤（CFO + 初相）

输出: sionna_dataset/sionna_amr_data.npz
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ╔══════════════════════════════════════════════════════════╗
# ║  1. 可调参数                                             ║
# ╚══════════════════════════════════════════════════════════╝
SIGNAL_LENGTH       = 1024
SPS                 = 4              # 每符号采样数
NUM_SYMBOLS         = SIGNAL_LENGTH // SPS   # 256
ROLLOFF             = 0.35
RRC_TAPS            = 101

N_PER_CLASS_PER_SNR = 200
SNR_LIST_DB         = list(range(-18, 22, 2))  # -18 ~ 20 dB

CHANNEL             = 'awgn'         # 'awgn' / 'rayleigh' / 'rician'
RICIAN_K_DB         = 10

ADD_IMPAIRMENTS     = True
MAX_CFO_NORM        = 0.001
MAX_PHASE           = 2 * np.pi

TRAIN_RATIO         = 0.8
SAVE_DIR            = './sionna_dataset'
os.makedirs(SAVE_DIR, exist_ok=True)

CLASSES = ['BPSK', 'QPSK', '8PSK', 'OQPSK', '2FSK', '4FSK',
           '16QAM', '2ASK', 'Pi4DQPSK', '64QAM', '32QAM',
           '128QAM', '256QAM', 'MSK']
NUM_CLASSES = len(CLASSES)


# ╔══════════════════════════════════════════════════════════╗
# ║  2. RRC 脉冲成形滤波器                                    ║
# ╚══════════════════════════════════════════════════════════╝
def design_rrc(ntaps, sps, beta):
    T = float(sps)
    idx = np.arange(-(ntaps // 2), ntaps // 2 + 1, dtype=np.float64)
    h = np.zeros_like(idx)
    for i, n in enumerate(idx):
        if abs(n) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(n) - T / (4.0 * beta)) < 1e-12:
            h[i] = beta / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * n / T * (1 - beta)) +
                   4 * beta * n / T * np.cos(np.pi * n / T * (1 + beta)))
            den = np.pi * n / T * (1 - (4 * beta * n / T) ** 2)
            h[i] = num / den
    return (h / np.linalg.norm(h)).astype(np.float64)


_rrc = design_rrc(RRC_TAPS, SPS, ROLLOFF)


def pulse_shape(symbols, sps=SPS, filt=_rrc, length=SIGNAL_LENGTH):
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols
    y = np.convolve(up, filt, mode='same')
    if len(y) >= length:
        return y[:length]
    return np.pad(y, (0, length - len(y)))


# ╔══════════════════════════════════════════════════════════╗
# ║  3. QAM / PSK 星座生成（替代 Sionna Mapper）              ║
# ╚══════════════════════════════════════════════════════════╝
def _gray_code(n_bits):
    """生成 n_bits 位格雷码序列"""
    if n_bits == 0:
        return [0]
    lower = _gray_code(n_bits - 1)
    return lower + [x + (1 << (n_bits - 1)) for x in reversed(lower)]


def _make_qam_constellation(M):
    """
    生成标准方形 QAM 星座 (M = 4, 16, 64, 256)
    对于 M = 8, 32, 128 用 Cross-QAM 或最近方形近似
    """
    bps = int(np.log2(M))

    if bps % 2 == 0:
        # 方形 QAM: sqrt(M) x sqrt(M)
        k = int(np.sqrt(M))
        vals = 2 * np.arange(k) - (k - 1)  # e.g. [-3,-1,1,3] for 16QAM
        grid_i, grid_q = np.meshgrid(vals, vals)
        pts = (grid_i + 1j * grid_q).flatten()
    else:
        # 非方形 (32QAM, 128QAM): 用最近方形然后截取
        k_up = int(np.ceil(np.sqrt(M)))
        if k_up % 2 == 1:
            k_up += 1
        vals = 2 * np.arange(k_up) - (k_up - 1)
        grid_i, grid_q = np.meshgrid(vals, vals)
        pts_all = (grid_i + 1j * grid_q).flatten()
        # 按功率排序，取前 M 个
        power = np.abs(pts_all) ** 2
        idx = np.argsort(power)[:M]
        pts = pts_all[idx]

    # 归一化到单位平均功率
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts


def _make_psk_constellation(M):
    """生成 M-PSK 星座"""
    pts = np.exp(1j * 2 * np.pi * np.arange(M) / M)
    return pts


# 预生成各阶星座
_QAM_CONST = {}
for _m in [4, 16, 32, 64, 128, 256]:
    _QAM_CONST[_m] = _make_qam_constellation(_m)

_PSK_CONST = {}
for _m in [2, 4, 8]:
    _PSK_CONST[_m] = _make_psk_constellation(_m)


def random_qam_symbols(M, n_batch, n_sym=NUM_SYMBOLS):
    """随机生成 M-QAM 符号 (n_batch, n_sym)"""
    const = _QAM_CONST[M]
    idx = np.random.randint(0, M, (n_batch, n_sym))
    return const[idx]


def random_psk_symbols(M, n_batch, n_sym=NUM_SYMBOLS):
    """随机生成 M-PSK 符号 (n_batch, n_sym)"""
    const = _PSK_CONST[M]
    idx = np.random.randint(0, M, (n_batch, n_sym))
    return const[idx]


# ╔══════════════════════════════════════════════════════════╗
# ║  4. 各调制方式生成器                                       ║
# ╚══════════════════════════════════════════════════════════╝

# ---------- A. PSK / QAM + RRC ----------
def gen_bpsk(n):
    syms = random_psk_symbols(2, n)
    return np.array([pulse_shape(s) for s in syms])

def gen_qpsk(n):
    syms = random_psk_symbols(4, n)
    return np.array([pulse_shape(s) for s in syms])

def gen_8psk(n):
    syms = random_psk_symbols(8, n)
    return np.array([pulse_shape(s) for s in syms])

def gen_16qam(n):
    syms = random_qam_symbols(16, n)
    return np.array([pulse_shape(s) for s in syms])

def gen_32qam(n):
    syms = random_qam_symbols(32, n)
    return np.array([pulse_shape(s) for s in syms])

def gen_64qam(n):
    syms = random_qam_symbols(64, n)
    return np.array([pulse_shape(s) for s in syms])

def gen_128qam(n):
    syms = random_qam_symbols(128, n)
    return np.array([pulse_shape(s) for s in syms])

def gen_256qam(n):
    syms = random_qam_symbols(256, n)
    return np.array([pulse_shape(s) for s in syms])


# ---------- B. 特殊线性调制 ----------
def gen_oqpsk(n):
    sigs = []
    for _ in range(n):
        bi = 2.0 * np.random.randint(0, 2, NUM_SYMBOLS) - 1.0
        bq = 2.0 * np.random.randint(0, 2, NUM_SYMBOLS) - 1.0
        up_i = np.zeros(NUM_SYMBOLS * SPS)
        up_i[::SPS] = bi
        si = np.convolve(up_i, _rrc, mode='same')
        up_q = np.zeros(NUM_SYMBOLS * SPS + SPS)
        for k in range(NUM_SYMBOLS):
            up_q[k * SPS + SPS // 2] = bq[k]
        sq = np.convolve(up_q[:NUM_SYMBOLS * SPS], _rrc, mode='same')
        sig = si[:SIGNAL_LENGTH] + 1j * sq[:SIGNAL_LENGTH]
        sigs.append(sig)
    return np.array(sigs)


def gen_pi4dqpsk(n):
    delta_phi = np.array([np.pi / 4, 3 * np.pi / 4,
                          -3 * np.pi / 4, -np.pi / 4])
    sigs = []
    for _ in range(n):
        idx = np.random.randint(0, 4, NUM_SYMBOLS)
        phase = np.cumsum(delta_phi[idx])
        syms = np.exp(1j * phase)
        sigs.append(pulse_shape(syms))
    return np.array(sigs)


def gen_2ask(n):
    sigs = []
    for _ in range(n):
        bits = np.random.randint(0, 2, NUM_SYMBOLS).astype(np.float64)
        sigs.append(pulse_shape(bits + 0j))
    return np.array(sigs)


# ---------- C. 频率调制 ----------
def gen_2fsk(n):
    h_mod = 1.0
    delta_f = h_mod / (2.0 * SPS)
    sigs = []
    for _ in range(n):
        bits = np.random.randint(0, 2, NUM_SYMBOLS)
        freq = np.zeros(SIGNAL_LENGTH)
        for k in range(NUM_SYMBOLS):
            freq[k * SPS:(k + 1) * SPS] = (2 * bits[k] - 1) * delta_f
        phase = 2 * np.pi * np.cumsum(freq)
        sigs.append(np.exp(1j * phase))
    return np.array(sigs)


def gen_4fsk(n):
    delta_f = 1.0 / (6.0 * SPS)
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    sigs = []
    for _ in range(n):
        idx = np.random.randint(0, 4, NUM_SYMBOLS)
        freq = np.zeros(SIGNAL_LENGTH)
        for k in range(NUM_SYMBOLS):
            freq[k * SPS:(k + 1) * SPS] = levels[idx[k]] * delta_f
        phase = 2 * np.pi * np.cumsum(freq)
        sigs.append(np.exp(1j * phase))
    return np.array(sigs)


def gen_msk(n):
    delta_f = 1.0 / (4.0 * SPS)
    sigs = []
    for _ in range(n):
        bits = np.random.randint(0, 2, NUM_SYMBOLS)
        freq = np.zeros(SIGNAL_LENGTH)
        for k in range(NUM_SYMBOLS):
            freq[k * SPS:(k + 1) * SPS] = (2 * bits[k] - 1) * delta_f
        phase = 2 * np.pi * np.cumsum(freq)
        sigs.append(np.exp(1j * phase))
    return np.array(sigs)


# ╔══════════════════════════════════════════════════════════╗
# ║  5. 信道模型（纯 NumPy）                                  ║
# ╚══════════════════════════════════════════════════════════╝
def apply_awgn(signals, snr_db):
    """
    AWGN 信道
    signals: (N, L) complex
    snr_db:  float
    """
    N, L = signals.shape
    # 功率归一化
    pwr = np.mean(np.abs(signals) ** 2, axis=-1, keepdims=True)
    x = signals / np.sqrt(pwr + 1e-12)

    # 噪声
    noise_power = 1.0 / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(N, L) + 1j * np.random.randn(N, L))
    return (x + noise).astype(np.complex64)


def apply_channel(signals, snr_db, channel_type=CHANNEL):
    N, L = signals.shape
    pwr = np.mean(np.abs(signals) ** 2, axis=-1, keepdims=True)
    x = signals / np.sqrt(pwr + 1e-12)

    # 衰落
    if channel_type == 'rayleigh':
        h = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)
        x = x * h
    elif channel_type == 'rician':
        K = 10 ** (RICIAN_K_DB / 10)
        h_los = np.sqrt(K / (K + 1))
        h_nlos = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) \
                 / np.sqrt(2) * np.sqrt(1 / (K + 1))
        x = x * (h_los + h_nlos)

    # AWGN
    noise_power = 1.0 / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(N, L) + 1j * np.random.randn(N, L))

    return (x + noise).astype(np.complex64)


# ╔══════════════════════════════════════════════════════════╗
# ║  6. 射频损伤                                              ║
# ╚══════════════════════════════════════════════════════════╝
def add_impairments(signals):
    N, L = signals.shape
    phi0 = np.random.uniform(0, MAX_PHASE, (N, 1))
    cfo  = np.random.uniform(-MAX_CFO_NORM, MAX_CFO_NORM, (N, 1))
    t    = np.arange(L).reshape(1, L)
    return signals * np.exp(1j * (2 * np.pi * cfo * t + phi0))


# ╔══════════════════════════════════════════════════════════╗
# ║  7. 生成器注册表                                          ║
# ╚══════════════════════════════════════════════════════════╝
GENERATORS = {
    'BPSK':     gen_bpsk,
    'QPSK':     gen_qpsk,
    '8PSK':     gen_8psk,
    'OQPSK':    gen_oqpsk,
    '2FSK':     gen_2fsk,
    '4FSK':     gen_4fsk,
    '16QAM':    gen_16qam,
    '2ASK':     gen_2ask,
    'Pi4DQPSK': gen_pi4dqpsk,
    '64QAM':    gen_64qam,
    '32QAM':    gen_32qam,
    '128QAM':   gen_128qam,
    '256QAM':   gen_256qam,
    'MSK':      gen_msk,
}


# ╔══════════════════════════════════════════════════════════╗
# ║  8. 主生成流程                                            ║
# ╚══════════════════════════════════════════════════════════╝
def generate_all():
    all_iq, all_y, all_snr = [], [], []
    total = NUM_CLASSES * len(SNR_LIST_DB)
    pbar = tqdm(total=total, desc='Generating', ncols=80)

    for cls_idx, cls_name in enumerate(CLASSES):
        gen_fn = GENERATORS[cls_name]

        for snr_db in SNR_LIST_DB:
            clean = gen_fn(N_PER_CLASS_PER_SNR)

            if ADD_IMPAIRMENTS:
                clean = add_impairments(clean)

            noisy = apply_channel(clean, snr_db)

            iq = np.stack([noisy.real, noisy.imag], axis=1).astype(np.float32)

            all_iq.append(iq)
            all_y.extend([cls_idx] * N_PER_CLASS_PER_SNR)
            all_snr.extend([float(snr_db)] * N_PER_CLASS_PER_SNR)

            pbar.set_postfix(cls=cls_name, snr=f'{snr_db}dB')
            pbar.update(1)

    pbar.close()

    X   = np.concatenate(all_iq, axis=0)
    Y   = np.array(all_y,   dtype=np.int64)
    SNR = np.array(all_snr, dtype=np.float32)

    print(f'\n总样本数: {len(Y):,}')
    print(f'  X shape : {X.shape}  dtype={X.dtype}')
    print(f'  Y shape : {Y.shape}  classes={np.unique(Y)}')
    print(f'  SNR范围 : {SNR.min():.0f} ~ {SNR.max():.0f} dB')
    return X, Y, SNR


def split_and_save(X, Y, SNR):
    idx_all = np.arange(len(Y))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=1 - TRAIN_RATIO,
        random_state=42, stratify=Y)

    save_path = os.path.join(SAVE_DIR, 'sionna_amr_data.npz')
    np.savez_compressed(
        save_path,
        X_train   = X[idx_train],
        Y_train   = Y[idx_train],
        X_test    = X[idx_test],
        Y_test    = Y[idx_test],
        snr_test  = SNR[idx_test],
        classes   = np.array(CLASSES),
        snr_list  = np.array(SNR_LIST_DB, dtype=np.float32),
    )

    print(f'\n已保存 → {save_path}')
    print(f'  Train : {len(idx_train):,}  ({X[idx_train].shape})')
    print(f'  Test  : {len(idx_test):,}   ({X[idx_test].shape})')

    sz = os.path.getsize(save_path) / 1024 / 1024
    print(f'  文件大小: {sz:.1f} MB')
    return save_path


def quick_verify(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    cls_names = list(data['classes'])

    print('\n===== 数据验证 =====')
    for split in ['train', 'test']:
        X = data[f'X_{split}']
        Y = data[f'Y_{split}']
        print(f'{split}: X={X.shape} Y={Y.shape}')
        for c in range(NUM_CLASSES):
            cnt = np.sum(Y == c)
            print(f'    [{c:2d}] {cls_names[c]:>10s}: {cnt:5d} samples')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 7, figsize=(21, 6))
        X_test = data['X_test']
        Y_test = data['Y_test']
        for c in range(NUM_CLASSES):
            ax = axes[c // 7, c % 7]
            mask = Y_test == c
            sample = X_test[mask][0]
            ax.plot(sample[0, :200], sample[1, :200], '.', ms=1, alpha=0.6)
            ax.set_title(cls_names[c], fontsize=9)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
        plt.suptitle('IQ Scatter (first 200 pts per sample)', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'verify_iq_scatter.png'),
                    dpi=150, bbox_inches='tight')
        print(f'验证图已保存 → {SAVE_DIR}/verify_iq_scatter.png')
    except Exception as e:
        print(f'画图跳过: {e}')


# ╔══════════════════════════════════════════════════════════╗
# ║  Main                                                    ║
# ╚══════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    print('=' * 60)
    print('AMR 仿真数据集生成器（纯 NumPy）')
    print(f'  调制类型: {NUM_CLASSES} 类')
    print(f'  SNR 范围: {SNR_LIST_DB[0]} ~ {SNR_LIST_DB[-1]} dB '
          f'({len(SNR_LIST_DB)} 档)')
    print(f'  每类每SNR: {N_PER_CLASS_PER_SNR} 条')
    print(f'  总样本数:  {NUM_CLASSES * len(SNR_LIST_DB) * N_PER_CLASS_PER_SNR:,}')
    print(f'  信道: {CHANNEL}')
    print(f'  SPS: {SPS}  信号长度: {SIGNAL_LENGTH}  符号数: {NUM_SYMBOLS}')
    print('=' * 60)

    X, Y, SNR = generate_all()
    path = split_and_save(X, Y, SNR)
    quick_verify(path)
    print('\n✓ 完成!')