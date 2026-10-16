"""
hybrid_features.py
==================
在内存中计算两类辅助特征，无需任何图片文件：

  Branch B — 瞬时特征 (Instantaneous features)
    shape: (N, 3, L)  — 与 IQ 时序同长，供 1-D CNN 处理
    通道 0: 瞬时幅度  A[n] = |I + jQ|
    通道 1: 瞬时频率  f[n] = dφ/dn / (2π)   (归一化)
    通道 2: 瞬时相位  φ[n] ∈ (-π, π]

  Branch C — 循环/统计特征向量 (Cyclic & higher-order statistics)
    shape: (N, FEAT_DIM)  — 固定维度标量向量，供 MLP/FC 处理
    包含：
      · 二阶循环自相关 C²α(0)  @ α=0, 0.5 (归一化)
      · 四阶循环累积量 C⁴α(0,0,0) @ α=0  (Mendel 定义)
      · 高阶矩: E[A²], E[A⁴], E[A⁶], E[A⁸]  (σ-归一化)
      · 相位统计: σ(φ), γ₁(φ), γ₂(φ)
      · 频率统计: σ(f), max(|f|)
      · IQ 相关: ρ(I,Q), E[I²Q²] / (E[I²]·E[Q²])

FEAT_DIM = 20  (固定，模型构建时使用)
"""

import numpy as np
import torch


# ── 公开常量 ───────────────────────────────────────────────────────────────
INST_CHANNELS = 3      # Branch B 通道数
FEAT_DIM      = 20     # Branch C 向量维度


# ══════════════════════════════════════════════════════════════════════════
# Branch B: 瞬时特征  (numpy, batch 矢量化)
# ══════════════════════════════════════════════════════════════════════════

def compute_instantaneous(iq: np.ndarray) -> np.ndarray:
    """
    输入: iq  shape (N, 2, L)  float32, 归一化后
    返回: out shape (N, 3, L)  float32
      ch0 = 瞬时幅度 (已功率归一化)
      ch1 = 瞬时频率 (弧度/样本 / π, ∈ [-1,1] 附近)
      ch2 = 瞬时相位 (/ π, ∈ (-1,1])
    """
    I = iq[:, 0, :]   # (N, L)
    Q = iq[:, 1, :]

    cplx = I + 1j * Q  # (N, L) complex128

    # 幅度
    A = np.abs(cplx)                               # (N, L)
    # 防止功率为 0
    pwr = np.mean(A ** 2, axis=1, keepdims=True)
    A_norm = A / np.sqrt(np.maximum(pwr, 1e-10))  # (N, L)

    # 瞬时相位
    phi = np.angle(cplx)                           # (N, L)  ∈ (-π, π]

    # 瞬时频率：解卷绕相位差分
    phi_unwrap = np.unwrap(phi, axis=1)
    freq = np.diff(phi_unwrap, axis=1, prepend=phi_unwrap[:, :1])  # (N, L)

    # 归一化到 [-1, 1] 附近
    freq_norm = freq / np.pi
    phi_norm  = phi  / np.pi

    out = np.stack([A_norm, freq_norm, phi_norm], axis=1).astype(np.float32)
    return out   # (N, 3, L)


# ══════════════════════════════════════════════════════════════════════════
# Branch C: 循环 & 高阶统计特征  (numpy, batch 矢量化)
# ══════════════════════════════════════════════════════════════════════════

def _safe_std(x, axis):
    return np.std(x, axis=axis) + 1e-8


def _cyclic_c2_alpha(cplx: np.ndarray, alpha_norm: float) -> np.ndarray:
    """
    归一化循环自相关（零延迟）
    C²α(0) = (1/L) Σ x[n]·x*[n] · exp(-j 2π α n)
    alpha_norm: α / fs (0=AWGN, 0.5=常见线性调制谱线)
    返回 |C²α(0)| / E[|x|²], shape (N,)
    """
    N, L = cplx.shape
    n = np.arange(L, dtype=np.float64)
    exp_vec = np.exp(-1j * 2 * np.pi * alpha_norm * n)  # (L,)
    # 批量
    rxx_alpha = np.mean(cplx * np.conj(cplx) * exp_vec[np.newaxis, :], axis=1)  # (N,)
    pwr = np.mean(np.abs(cplx) ** 2, axis=1) + 1e-10
    return np.abs(rxx_alpha) / pwr   # (N,)


def _cyclic_c4_alpha0(cplx: np.ndarray) -> np.ndarray:
    """
    四阶循环累积量（α=0, τ1=τ2=τ3=0）:
    C⁴(0) = E[x⁴] - 3·(E[x²])²
    使用复数矩：以幅度序列计算更稳定
    返回 |C⁴(0)| / (E[|x|²])², shape (N,)
    """
    pwr = np.mean(np.abs(cplx) ** 2, axis=1) + 1e-10
    m2  = pwr
    m4  = np.mean(np.abs(cplx) ** 4, axis=1)
    c4  = np.abs(m4 - 2.0 * m2 ** 2)   # 简化实数版本
    return c4 / (m2 ** 2 + 1e-10)


def compute_cyclic_features(iq: np.ndarray) -> np.ndarray:
    """
    输入: iq  shape (N, 2, L)  float32
    返回: feat shape (N, FEAT_DIM)  float32

    特征列表 (20 维):
      0   C²α=0(0)      — 直流处循环自相关（功率比）
      1   C²α=0.5(0)    — 符号速率谱线（对 PSK/QAM 敏感）
      2   C⁴α=0(0,0,0)  — 四阶累积量（区分 BPSK/QPSK/16QAM 等）
      3   E[A²]         — 二阶幅度矩（归一化）
      4   E[A⁴]/E[A²]²  — 四阶幅度矩（调制指数）
      5   E[A⁶]/E[A²]³  — 六阶幅度矩
      6   E[A⁸]/E[A²]⁴  — 八阶幅度矩
      7   σ(A)          — 幅度标准差
      8   σ(φ)          — 相位标准差
      9   γ₁(φ)         — 相位偏度
      10  γ₂(φ)         — 相位峰度 - 3
      11  σ(f)          — 频率标准差
      12  max|f| / π    — 频率最大绝对值（FSK 敏感）
      13  ρ(I,Q)        — IQ 相关系数（OQPSK/Pi4DQPSK 敏感）
      14  E[I²Q²]/(E[I²]·E[Q²]) — IQ 二阶交叉矩比
      15  E[A] / sqrt(E[A²])    — 幅度平均/RMS（AM 敏感）
      16  max(A) / RMS          — 峰值因子 (PAPR proxy)
      17  std(A²) / E[A²]       — 幅度方差比
      18  C²α=0.25(0)           — 额外循环谱线
      19  spectral_entropy      — 归一化频谱熵
    """
    N, _, L = iq.shape
    I = iq[:, 0, :].astype(np.float64)
    Q = iq[:, 1, :].astype(np.float64)
    cplx = I + 1j * Q

    # 幅度 & 相位 & 频率
    A   = np.abs(cplx)                                      # (N, L)
    phi = np.angle(cplx)                                    # (N, L)
    phi_uw = np.unwrap(phi, axis=1)
    freq    = np.diff(phi_uw, axis=1, prepend=phi_uw[:, :1]) / np.pi  # (N, L)

    pwr  = np.mean(A ** 2, axis=1) + 1e-10                 # (N,)
    rms  = np.sqrt(pwr)

    feats = np.zeros((N, FEAT_DIM), dtype=np.float64)

    # 0-2: 循环特征
    feats[:, 0]  = _cyclic_c2_alpha(cplx, alpha_norm=0.0)
    feats[:, 1]  = _cyclic_c2_alpha(cplx, alpha_norm=0.5)
    feats[:, 2]  = _cyclic_c4_alpha0(cplx)

    # 3-7: 幅度高阶矩
    m2 = pwr
    m4 = np.mean(A ** 4, axis=1)
    m6 = np.mean(A ** 6, axis=1)
    m8 = np.mean(A ** 8, axis=1)
    feats[:, 3]  = m2 / (rms ** 2 + 1e-10)                 # ≈ 1
    feats[:, 4]  = m4 / (m2 ** 2 + 1e-10)                  # QAM 阶数敏感
    feats[:, 5]  = m6 / (m2 ** 3 + 1e-10)
    feats[:, 6]  = m8 / (m2 ** 4 + 1e-10)
    feats[:, 7]  = np.std(A, axis=1) / (rms + 1e-10)

    # 8-10: 相位统计
    phi_std = np.std(phi, axis=1)
    phi_mean = np.mean(phi, axis=1, keepdims=True)
    phi_c = phi - phi_mean
    feats[:, 8]  = phi_std
    sigma_phi = phi_std + 1e-8
    feats[:, 9]  = np.mean(phi_c ** 3, axis=1) / sigma_phi ** 3   # skewness
    feats[:, 10] = np.mean(phi_c ** 4, axis=1) / sigma_phi ** 4 - 3.0  # excess kurtosis

    # 11-12: 频率统计
    feats[:, 11] = np.std(freq, axis=1)
    feats[:, 12] = np.max(np.abs(freq), axis=1)

    # 13-14: IQ 交叉统计
    ei2  = np.mean(I ** 2, axis=1) + 1e-10
    eq2  = np.mean(Q ** 2, axis=1) + 1e-10
    std_i = np.std(I, axis=1) + 1e-8
    std_q = np.std(Q, axis=1) + 1e-8
    feats[:, 13] = np.mean(I * Q, axis=1) / (std_i * std_q)
    feats[:, 14] = np.mean(I ** 2 * Q ** 2, axis=1) / (ei2 * eq2)

    # 15-17: 幅度分布特征
    feats[:, 15] = np.mean(A, axis=1) / (rms + 1e-10)
    feats[:, 16] = np.max(A, axis=1) / (rms + 1e-10)
    feats[:, 17] = np.std(A ** 2, axis=1) / (m2 + 1e-10)

    # 18: 额外循环谱线 α=0.25
    feats[:, 18] = _cyclic_c2_alpha(cplx, alpha_norm=0.25)

    # 19: 归一化频谱熵 (PSD via FFT)
    NFFT = min(256, L)
    spec = np.abs(np.fft.fft(cplx[:, :NFFT], axis=1)) ** 2   # (N, NFFT)
    spec = spec / (spec.sum(axis=1, keepdims=True) + 1e-10)
    entropy = -np.sum(spec * np.log(spec + 1e-10), axis=1)
    entropy_max = np.log(NFFT + 1e-10)
    feats[:, 19] = entropy / entropy_max

    # 裁剪异常值
    feats = np.clip(feats, -30.0, 30.0)

    return feats.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# 统一接口：IQ → (inst_feat, cyclic_feat)
# ══════════════════════════════════════════════════════════════════════════

def extract_all(iq: np.ndarray):
    """
    输入: iq  (N, 2, L)  float32  (已功率归一化)
    返回:
      inst_feat   (N, 3, L)   float32  → Branch B
      cyclic_feat (N, FEAT_DIM) float32 → Branch C
    """
    inst   = compute_instantaneous(iq)
    cyclic = compute_cyclic_features(iq)
    return inst, cyclic


# ══════════════════════════════════════════════════════════════════════════
# Torch Dataset 包装 (懒计算，避免把所有特征提前写入内存)
# ══════════════════════════════════════════════════════════════════════════

class HybridDataset(torch.utils.data.Dataset):
    """
    将原始 IQ 数据集包装成三分支输入。
    特征在 __getitem__ 中逐样本计算，节省内存。

    返回: (iq_tensor, inst_tensor, cyclic_tensor, label)
      iq_tensor     (1024, 2)    — 原始 IQ (Branch A)
      inst_tensor   (3, 1024)    — 瞬时特征 (Branch B)
      cyclic_tensor (FEAT_DIM,)  — 循环统计 (Branch C)
      label         long scalar
    """

    def __init__(self, iq_data: np.ndarray, labels: np.ndarray,
                 normalize: bool = True, augment: bool = False):
        """
        iq_data: (N, 2, L)  float32
        labels:  (N,)       int64
        """
        self.raw   = iq_data
        self.labels = torch.from_numpy(labels).long()
        self.normalize = normalize
        self.augment   = augment

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _power_normalize(iq2L):
        """iq2L: (2, L)"""
        pwr = np.mean(iq2L[0] ** 2 + iq2L[1] ** 2)
        if pwr < 1e-10:
            return iq2L.astype(np.float32)
        return (iq2L / np.sqrt(pwr)).astype(np.float32)

    @staticmethod
    def _augment_iq(iq2L):
        """iq2L: (2, L) numpy"""
        # 随机时移
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            iq2L = np.roll(iq2L, shift, axis=1)
        # 加噪
        if np.random.random() < 0.5:
            sig_pwr = np.mean(iq2L ** 2)
            if sig_pwr > 1e-10:
                snr_db = np.random.uniform(10, 30)
                n_pwr = sig_pwr / (10 ** (snr_db / 10))
                iq2L = iq2L + np.random.normal(0, np.sqrt(n_pwr), iq2L.shape).astype(np.float32)
        # 随机相位旋转
        if np.random.random() < 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            I, Q = iq2L[0].copy(), iq2L[1].copy()
            iq2L[0] = I * np.cos(theta) - Q * np.sin(theta)
            iq2L[1] = I * np.sin(theta) + Q * np.cos(theta)
        return iq2L

    def __getitem__(self, idx):
        sample = self.raw[idx].copy()   # (2, L)

        if self.augment:
            sample = self._augment_iq(sample)

        if self.normalize:
            sample = self._power_normalize(sample)

        # Branch A: (L, 2)
        iq_tensor = torch.from_numpy(sample).float().permute(1, 0)

        # Branch B & C: 在 sample (2, L) 上计算
        sample_N = sample[np.newaxis, ...]   # (1, 2, L)
        inst_np, cyc_np = extract_all(sample_N)
        inst_tensor   = torch.from_numpy(inst_np[0]).float()    # (3, L)
        cyclic_tensor = torch.from_numpy(cyc_np[0]).float()     # (FEAT_DIM,)

        return iq_tensor, inst_tensor, cyclic_tensor, self.labels[idx]
