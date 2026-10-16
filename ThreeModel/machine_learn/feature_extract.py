"""
feature_extract.py
==================
将原始 IQ 信号 (N, 2, L) 提取为 8 通道特征张量 (N, 8, L)。

全程纯 numpy，无图片文件，无 OpenCV。
特征均按样本独立归一化，输出 float32。

通道定义
--------
Ch 0  — I 分量（原始，功率归一化后）
Ch 1  — Q 分量（原始，功率归一化后）
Ch 2  — 瞬时幅度  A[n] = |I+jQ|            （均值归一化）
Ch 3  — 瞬时相位  φ[n] = angle(I+jQ)/π     ∈(-1,1]
Ch 4  — 瞬时频率  f[n] = Δunwrap(φ)/π      (∈[-1,1] 附近)
Ch 5  — 幅度平方  A²[n]                      （均值归一化）
Ch 6  — 短时自相关  Re{x[n]·conj(x[n-k])}  k=SPS（默认4）
Ch 7  — 差分幅度   ΔA[n] = A[n] - A[n-1]   （std 归一化）

选择依据（针对 QAM 阶数区分）
------------------------------
· Ch 0-1: 保留原始 IQ，不损失任何信息
· Ch 2,5: 幅度及幅度平方——高阶 QAM 幅度环数更多，分布截然不同
· Ch 3:   相位——PSK/FSK/QAM 相位分布结构不同
· Ch 4:   瞬时频率——对 FSK/MSK 最敏感，辅助 QAM 内部区分
· Ch 6:   自相关——反映符号速率周期性结构（码间干扰/成型滤波余晖）
· Ch 7:   差分幅度——对幅度跳变的灵敏度，区分 ASK/高阶 QAM
"""

import numpy as np


# ── 公开常量 ──────────────────────────────────────────────────────────────
N_CHANNELS = 8    # 总通道数，模型构建时引用此常量


# ══════════════════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════════════════

def _safe_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """除以沿 axis 方向的 std，防止除零"""
    return x / (x.std(axis=axis, keepdims=True) + eps)


def _mean_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """减均值后除以 std"""
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True) + eps
    return (x - m) / s


# ══════════════════════════════════════════════════════════════════════════
# 主提取函数
# ══════════════════════════════════════════════════════════════════════════

def extract_features(iq: np.ndarray, sps: int = 4) -> np.ndarray:
    """
    Parameters
    ----------
    iq  : (N, 2, L)  float32/64，已功率归一化
    sps : 每符号采样数，用于自相关滞后 k=sps

    Returns
    -------
    feat : (N, N_CHANNELS, L)  float32
    """
    if iq.ndim == 2:           # 单样本兼容: (2, L) → (1, 2, L)
        iq = iq[np.newaxis]

    N, _, L = iq.shape
    I = iq[:, 0, :].astype(np.float64)   # (N, L)
    Q = iq[:, 1, :].astype(np.float64)
    cplx = I + 1j * Q                    # (N, L) complex128

    feat = np.empty((N, N_CHANNELS, L), dtype=np.float32)

    # ── Ch 0-1: 原始 IQ（已功率归一化，直接使用） ────────────────────────
    feat[:, 0, :] = I.astype(np.float32)
    feat[:, 1, :] = Q.astype(np.float32)

    # ── Ch 2: 瞬时幅度（均值归一化） ────────────────────────────────────
    A = np.abs(cplx)                              # (N, L)
    feat[:, 2, :] = _mean_norm(A).astype(np.float32)

    # ── Ch 3: 瞬时相位 /π ────────────────────────────────────────────────
    phi = np.angle(cplx) / np.pi                  # ∈(-1,1]
    feat[:, 3, :] = phi.astype(np.float32)

    # ── Ch 4: 瞬时频率（解卷绕相位一阶差分 /π） ─────────────────────────
    phi_uw = np.unwrap(np.angle(cplx), axis=1)
    freq   = np.diff(phi_uw, axis=1, prepend=phi_uw[:, :1]) / np.pi
    # 裁剪极端值再归一化
    freq   = np.clip(freq, -3.0, 3.0)
    feat[:, 4, :] = _mean_norm(freq).astype(np.float32)

    # ── Ch 5: 幅度平方（均值归一化） ────────────────────────────────────
    A2 = A ** 2
    feat[:, 5, :] = _mean_norm(A2).astype(np.float32)

    # ── Ch 6: 短时自相关 Re{x[n]·conj(x[n-k])}，k=sps ──────────────────
    k   = max(1, sps)
    pad = np.zeros((N, k), dtype=np.complex128)
    x_delayed = np.concatenate([pad, cplx[:, :-k]], axis=1)   # (N, L)
    autocorr  = np.real(cplx * np.conj(x_delayed))
    feat[:, 6, :] = _mean_norm(autocorr).astype(np.float32)

    # ── Ch 7: 差分幅度 ΔA[n] = A[n] - A[n-1]（std 归一化） ─────────────
    dA = np.diff(A, axis=1, prepend=A[:, :1])
    feat[:, 7, :] = _safe_norm(dA).astype(np.float32)

    return feat   # (N, 8, L)


# ══════════════════════════════════════════════════════════════════════════
# Dataset 包装
# ══════════════════════════════════════════════════════════════════════════

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    懒加载版：特征在 __getitem__ 中逐样本提取，内存占用低。

    若样本数较少或 CPU 资源充足，可改用 EagerFeatureDataset 一次性
    预提取全部特征（训练速度更快，内存占用更高）。

    返回: (feat_tensor, label)
      feat_tensor : (N_CHANNELS, L)  float32
      label       : long scalar
    """

    def __init__(self, iq_data: np.ndarray, labels: np.ndarray,
                 normalize: bool = True, augment: bool = False,
                 sps: int = 4):
        """
        iq_data : (N, 2, L)  float32
        labels  : (N,)       int64/int32
        """
        self.raw      = iq_data
        self.labels   = torch.from_numpy(np.array(labels)).long()
        self.normalize = normalize
        self.augment   = augment
        self.sps       = sps

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _power_normalize(iq2L: np.ndarray) -> np.ndarray:
        pwr = np.mean(iq2L[0] ** 2 + iq2L[1] ** 2)
        if pwr < 1e-10:
            return iq2L.astype(np.float32)
        return (iq2L / np.sqrt(pwr)).astype(np.float32)

    def _augment(self, iq2L: np.ndarray) -> np.ndarray:
        # 随机时移
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            iq2L = np.roll(iq2L, shift, axis=1)
        # 加轻噪
        if np.random.random() < 0.4:
            sig_pwr = np.mean(iq2L ** 2)
            if sig_pwr > 1e-10:
                snr_db = np.random.uniform(15, 35)
                n_pwr  = sig_pwr / (10 ** (snr_db / 10))
                iq2L   = iq2L + np.random.normal(0, np.sqrt(n_pwr),
                                                  iq2L.shape).astype(np.float32)
        # 随机相位旋转
        if np.random.random() < 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            I, Q  = iq2L[0].copy(), iq2L[1].copy()
            iq2L[0] = I * np.cos(theta) - Q * np.sin(theta)
            iq2L[1] = I * np.sin(theta) + Q * np.cos(theta)
        return iq2L

    def __getitem__(self, idx):
        sample = self.raw[idx].copy()         # (2, L)
        if self.augment:
            sample = self._augment(sample)
        if self.normalize:
            sample = self._power_normalize(sample)
        feat = extract_features(sample, sps=self.sps)   # (1, 8, L)
        feat_t = torch.from_numpy(feat[0]).float()       # (8, L)
        return feat_t, self.labels[idx]


class EagerFeatureDataset(Dataset):
    """
    预提取版：构造时一次性处理全部数据，__getitem__ 仅做 Tensor 转换。
    训练 DataLoader 速度快，适合数据集能放入内存时使用。
    """

    def __init__(self, iq_data: np.ndarray, labels: np.ndarray,
                 normalize: bool = True, sps: int = 4):
        N = len(labels)
        # 功率归一化
        if normalize:
            pwr = np.mean(iq_data[:, 0] ** 2 + iq_data[:, 1] ** 2,
                          axis=-1, keepdims=True)[:, :, np.newaxis]  # (N,1,1)
            # reshape 适配 (N,2,L)
            pwr2 = np.mean(iq_data ** 2, axis=(1, 2), keepdims=True)    # (N,1,1)
            safe_pwr = np.sqrt(np.maximum(pwr2, 1e-10))
            iq_norm = (iq_data / safe_pwr).astype(np.float32)
        else:
            iq_norm = iq_data.astype(np.float32)

        # 批量提取，分批以控制内存峰值
        chunk = 4096
        feat_list = []
        for i in range(0, N, chunk):
            feat_list.append(extract_features(iq_norm[i:i+chunk], sps=sps))
        self.feats  = torch.from_numpy(np.concatenate(feat_list, axis=0))  # (N,8,L)
        self.labels = torch.from_numpy(np.array(labels)).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx].float(), self.labels[idx]
