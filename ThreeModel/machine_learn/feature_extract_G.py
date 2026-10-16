"""
feature_extract.py
==================
将原始 IQ 信号 (N, 2, L) 提取为 8 通道特征张量 (N, 8, L)。
"""

import numpy as np


# ── 公开常量 ──────────────────────────────────────────────────────────────
N_CHANNELS = 8    # 总通道数，模型构建时引用此常量


# ══════════════════════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════════════════════

def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    差分特征处理：由于差分幅度均值理论上接近 0，直接返回以保留其方差分布差异。
    """
    return x


def _mean_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    中心化去直流：仅减去每个样本自身的均值，不除以标准差。
    """
    m = x.mean(axis=axis, keepdims=True)
    return x - m


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

    # ── Ch 0-1: 原始 IQ（直接使用） ────────────────────────────────────
    feat[:, 0, :] = I.astype(np.float32)
    feat[:, 1, :] = Q.astype(np.float32)

    # ── Ch 2: 瞬时幅度（仅去直流） ────────────────────────────────────
    A = np.abs(cplx)                              # (N, L)
    feat[:, 2, :] = _mean_norm(A).astype(np.float32)

    # ── Ch 3: 瞬时相位 /π ────────────────────────────────────────────────
    phi = np.angle(cplx) / np.pi                  # ∈(-1,1]
    feat[:, 3, :] = phi.astype(np.float32)

    # ── Ch 4: 瞬时频率（去直流） ────────────────────────────────────────
    phi_uw = np.unwrap(np.angle(cplx), axis=1)
    freq   = np.diff(phi_uw, axis=1, prepend=phi_uw[:, :1]) / np.pi
    freq   = np.clip(freq, -3.0, 3.0)
    feat[:, 4, :] = _mean_norm(freq).astype(np.float32)

    # ── Ch 5: 幅度平方（仅去直流） ────────────────────────────────────
    A2 = A ** 2
    feat[:, 5, :] = _mean_norm(A2).astype(np.float32)

    # ── Ch 6: 短时自相关 ──────────────────────────────────────────────
    k   = max(1, sps)
    pad = np.zeros((N, k), dtype=np.complex128)
    x_delayed = np.concatenate([pad, cplx[:, :-k]], axis=1)   # (N, L)
    autocorr  = np.real(cplx * np.conj(x_delayed))
    feat[:, 6, :] = _mean_norm(autocorr).astype(np.float32)

    # ── Ch 7: 差分幅度 ────────────────────────────────────────────────
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
    懒加载版：总归一化功率调整为 1.0
    """

    def __init__(self, iq_data: np.ndarray, labels: np.ndarray,
                 normalize: bool = True, augment: bool = False,
                 sps: int = 4):
        self.raw      = iq_data
        self.labels   = torch.from_numpy(np.array(labels)).long()
        self.normalize = normalize
        self.augment   = augment
        self.sps       = sps

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _power_normalize(iq2L: np.ndarray) -> np.ndarray:
        # 计算整帧平均总功率 E[I^2 + Q^2]
        pwr = np.mean(iq2L[0] ** 2 + iq2L[1] ** 2)
        if pwr < 1e-10:
            return iq2L.astype(np.float32)
        return (iq2L / np.sqrt(pwr)).astype(np.float32)

    def _augment(self, iq2L: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            iq2L = np.roll(iq2L, shift, axis=1)
        if np.random.random() < 0.4:
            sig_pwr = np.mean(iq2L ** 2)
            if sig_pwr > 1e-10:
                snr_db = np.random.uniform(15, 35)
                n_pwr  = sig_pwr / (10 ** (snr_db / 10))
                iq2L   = iq2L + np.random.normal(0, np.sqrt(n_pwr),
                                                  iq2L.shape).astype(np.float32)
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
    预提取版：修复了原先归一化功率偏大（偏向2.0）的偏差，统一调整为 1.0 标准
    """

    def __init__(self, iq_data: np.ndarray, labels: np.ndarray,
                 normalize: bool = True, sps: int = 4):
        N = len(labels)
        if normalize:
            # 统一归一化标准：计算每个样本在 L 维度上的总功率
            pwr = np.mean(iq_data[:, 0, :] ** 2 + iq_data[:, 1, :] ** 2, axis=-1, keepdims=True)  # (N, 1)
            pwr = pwr[:, :, np.newaxis]  # 扩展为 (N, 1, 1) 以便广播
            safe_pwr = np.sqrt(np.maximum(pwr, 1e-10))
            iq_norm = (iq_data / safe_pwr).astype(np.float32)
        else:
            iq_norm = iq_data.astype(np.float32)

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