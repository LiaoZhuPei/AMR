"""
SDAE-LOF 项目 —— 特征维度可选(32/64/128) + 纯LOF检测器
========================================================================
数据流程:
  原始CSV -> 加载(丢弃第1025列, 文件夹名为标签) -> 特征提取(32/64/128维可选) -> 标准化
  -> 训练集(仅WiFi+蓝牙正常信号的一部分)  训练LOF + 基于训练集分布确定阈值
  -> 测试集(正常剩余部分 + 全部无人机)    最终评估，无人机数据全程不参与训练/调参
"""

import os
import glob
import numpy as np
import pandas as pd
import pywt
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

from pyod.models.lof import LOF

import joblib


# =========================================================
# 0. 全局配置
# =========================================================
class Config:
    DATA_ROOT = "./data_root"
    FS = 1.0                        # 采样率，若已知真实值请修改
    RANDOM_STATE = 42

    # ---- 数据划分 ----
    TEST_SIZE = 0.2                 # 正常样本中划出多少做测试，其余全部用于训练

    # ---- 特征维度选择: 32 / 64 / 128 ----
    FEATURE_DIM = 128

    # ---- 异常比例先验 (仅基于训练集自身分布确定阈值，不使用任何无人机数据) ----
    CONTAMINATION = 0.1

    # ---- LOF 超参数 ----
    LOF_N_NEIGHBORS = 20
    LOF_METRIC = "minkowski"

    WIFI_DIRS = ["TPLINK_TL_WR940N", "CISCO_LINKSYS_E3200"]
    BT_DIRS = ["APPLE_IPAD3", "APPLE_IPHONE6S", "APPLE_IPHONE7",
               "FITBIT_CHARGE3", "MOTOROLA"]
    DRONE_DIRS = ["DJI_M600", "BEEBEERUN", "DJI_INSPIRE",
                  "DJI_MAVICPRO", "DJI_PHANTOM"]


def set_seed(seed):
    np.random.seed(seed)


# =========================================================
# 1. 数据加载 (丢弃第1025列, 文件夹名为标签)
# =========================================================
def load_folder_csv(folder_path):
    all_rows = []
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if len(csv_files) == 0:
        print(f"[警告] 文件夹 {folder_path} 未找到 csv 文件")
    for f in csv_files:
        df = pd.read_csv(f, header=None)   # 若有表头改 header=0
        data = df.iloc[:, :-1].values.astype(np.float64)   # 丢弃第1025列
        all_rows.append(data)
    if len(all_rows) == 0:
        return np.empty((0, 1024), dtype=np.float64)
    return np.concatenate(all_rows, axis=0)


def load_dataset(data_root, dir_list):
    X_list, label_list = [], []
    for d in dir_list:
        folder = os.path.join(data_root, d)
        X = load_folder_csv(folder)
        if X.shape[0] == 0:
            continue
        X_list.append(X)
        label_list.extend([d] * X.shape[0])
    if len(X_list) == 0:
        return np.empty((0, 1024), dtype=np.float64), np.array([])
    return np.concatenate(X_list, axis=0), np.array(label_list)


# =========================================================
# 2. 特征提取: 32 / 64 / 128 维 三档可选
# =========================================================

# ---------------------------------------------------------
# 2.1 基础32维特征 (时域8 + 频域12 + STFT6 + 累积量2 + 包络4)
# ---------------------------------------------------------
def extract_32_features(x, fs=1.0):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    eps = 1e-12

    mean_val = np.mean(x)
    std_val = np.std(x)
    rms_val = np.sqrt(np.mean(x ** 2))
    p2p = np.max(x) - np.min(x)
    peak = np.max(np.abs(x))
    crest_factor = peak / (rms_val + eps)
    skewness = skew(x)
    kurt = kurtosis(x)
    zc = np.sum(np.abs(np.diff(np.sign(x))) > 0)
    zcr = zc / (N - 1)

    X = fft(x)
    half = N // 2
    mag = np.abs(X[:half])
    freqs = fftfreq(N, d=1.0 / fs)[:half]
    power = mag ** 2
    power_sum = np.sum(power) + eps

    spectral_centroid = np.sum(freqs * power) / power_sum
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / power_sum)
    spectral_skew = np.sum(((freqs - spectral_centroid) ** 3) * power) / (
            power_sum * (spectral_spread ** 3 + eps))
    spectral_kurt = np.sum(((freqs - spectral_centroid) ** 4) * power) / (
            power_sum * (spectral_spread ** 4 + eps)) - 3

    peak_idx = np.argmax(mag)
    peak_freq = freqs[peak_idx]
    peak_mag = mag[peak_idx]

    def bandwidth_at_db(mag, freqs, peak_mag, db):
        threshold = peak_mag / (10 ** (db / 20.0))
        idx = np.where(mag >= threshold)[0]
        if len(idx) == 0:
            return 0.0
        return freqs[idx[-1]] - freqs[idx[0]]

    bw_3db = bandwidth_at_db(mag, freqs, peak_mag, 3)
    bw_10db = bandwidth_at_db(mag, freqs, peak_mag, 10)

    psd_norm = power / power_sum
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + eps)) / np.log2(len(psd_norm) + eps)
    spectral_flatness = np.exp(np.mean(np.log(power + eps))) / (np.mean(power) + eps)

    cumsum_power = np.cumsum(power)
    idx_85 = min(np.searchsorted(cumsum_power, 0.85 * power_sum), len(freqs) - 1)
    idx_95 = min(np.searchsorted(cumsum_power, 0.95 * power_sum), len(freqs) - 1)
    rolloff_85 = freqs[idx_85]
    rolloff_95 = freqs[idx_95]

    log_mag = np.log(mag + eps)
    slope, _ = np.polyfit(freqs, log_mag, 1)
    spectral_slope = slope

    nperseg = min(256, N)
    f_stft, t_stft, Zxx = sp_signal.stft(x, fs=fs, nperseg=nperseg,
                                          noverlap=nperseg // 2)
    stft_mag = np.abs(Zxx)
    frame_energy = np.sum(stft_mag ** 2, axis=0)
    stft_energy_mean = np.mean(frame_energy)
    stft_energy_std = np.std(frame_energy)

    frame_power_sum = np.sum(stft_mag ** 2, axis=0) + eps
    frame_centroids = np.sum(f_stft[:, None] * (stft_mag ** 2), axis=0) / frame_power_sum
    freq_centroid_mean = np.mean(frame_centroids)
    freq_centroid_std = np.std(frame_centroids)

    if stft_mag.shape[1] > 1:
        flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
        flux_mean = np.mean(flux)
        flux_std = np.std(flux)
    else:
        flux_mean, flux_std = 0.0, 0.0

    analytic = sp_signal.hilbert(x)
    z = analytic
    M20 = np.mean(z ** 2)
    M21 = np.mean(np.abs(z) ** 2)
    M40 = np.mean(z ** 4)
    M42 = np.mean(np.abs(z) ** 4)

    C40 = np.abs(M40 - 3 * (M20 ** 2))
    C42 = np.abs(M42 - np.abs(M20) ** 2 - 2 * (M21 ** 2))

    envelope = np.abs(analytic)
    envelope_mean = np.mean(envelope)
    envelope_std = np.std(envelope)
    envelope_kurt = kurtosis(envelope)

    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    inst_freq_std = np.std(inst_freq)

    features = [
        mean_val, std_val, rms_val, p2p, crest_factor,
        skewness, kurt, zcr,
        spectral_centroid, spectral_spread, spectral_skew, spectral_kurt,
        peak_freq, bw_3db, bw_10db,
        spectral_entropy, spectral_flatness, rolloff_85, rolloff_95, spectral_slope,
        stft_energy_mean, stft_energy_std,
        freq_centroid_mean, freq_centroid_std,
        flux_mean, flux_std,
        C40, C42,
        envelope_mean, envelope_std, envelope_kurt,
        inst_freq_std
    ]
    return features, (X, freqs, mag, power, analytic, envelope, phase, inst_freq)


FEATURE_NAMES_32 = [
    "Mean", "Std", "RMS", "Peak-to-Peak", "Crest Factor",
    "Skewness", "Kurtosis", "Zero Crossing Rate",
    "Spectral Centroid", "Spectral Spread", "Spectral Skewness", "Spectral Kurtosis",
    "Peak Frequency", "-3dB Bandwidth", "-10dB Bandwidth",
    "Spectral Entropy", "Spectral Flatness", "Roll-off 85%", "Roll-off 95%", "Spectral Slope",
    "STFT Energy Mean", "STFT Energy Std",
    "Frequency Centroid Mean", "Frequency Centroid Std",
    "Spectral Flux Mean", "Spectral Flux Std",
    "C40", "C42",
    "Envelope Mean", "Envelope Std", "Envelope Kurtosis",
    "Instantaneous Frequency Std"
]


# ---------------------------------------------------------
# 2.2 64维增量特征 (在32维基础上新增32个)
# ---------------------------------------------------------
def extract_64_extra_features(x, fs, cached):
    """
    在32维特征基础上追加32个特征，复用cached中间量避免重复计算
    新增类别: 时域补充(6) + 频域补充(9) + STFT补充(9) + 包络/瞬时频率补充(8)
    """
    X, freqs, mag, power, analytic, envelope, phase, inst_freq = cached
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    eps = 1e-12
    power_sum = np.sum(power) + eps

    # ---- 时域补充 (6) ----
    median_val = np.median(x)
    var_val = np.var(x)
    mav_val = np.mean(np.abs(x))
    mad_val = np.median(np.abs(x - median_val))
    rms_val = np.sqrt(np.mean(x ** 2))
    shape_factor = rms_val / (mav_val + eps)
    impulse_factor = np.max(np.abs(x)) / (mav_val + eps)

    time_extra = [median_val, var_val, mav_val, mad_val, shape_factor, impulse_factor]

    # ---- 频域补充 (9) ----
    spectral_variance = np.var(freqs, ) if False else None  # 占位，用真实spread平方替代
    spectral_centroid = np.sum(freqs * power) / power_sum
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / power_sum)
    spectral_variance = spectral_spread ** 2

    peak_mag = np.max(mag)
    spectral_crest_factor = peak_mag / (np.mean(mag) + eps)

    k_idx = np.arange(1, len(mag))
    spectral_decrease = np.sum((mag[1:] - mag[0]) / k_idx) / (np.sum(mag[1:]) + eps)

    cumsum_power = np.cumsum(power)
    idx_50 = min(np.searchsorted(cumsum_power, 0.50 * power_sum), len(freqs) - 1)
    rolloff_50 = freqs[idx_50]

    def bandwidth_at_db(mag, freqs, peak_mag, db):
        threshold = peak_mag / (10 ** (db / 20.0))
        idx = np.where(mag >= threshold)[0]
        if len(idx) == 0:
            return 0.0
        return freqs[idx[-1]] - freqs[idx[0]]

    bw_6db = bandwidth_at_db(mag, freqs, peak_mag, 6)

    spectral_energy = power_sum
    idx_band3 = np.where(mag >= peak_mag / (10 ** (3 / 20.0)))[0]
    band_energy = np.sum(power[idx_band3]) if len(idx_band3) > 0 else 0.0
    spectral_energy_ratio = band_energy / power_sum
    spectral_papr = (peak_mag ** 2) / (np.mean(power) + eps)

    freq_extra = [spectral_variance, spectral_crest_factor, spectral_decrease,
                  rolloff_50, bw_6db, spectral_energy, spectral_energy_ratio,
                  spectral_papr, 0.0]  # 占位补齐到9个，最后一位留给频谱二阶差分能量
    # 频谱二阶差分能量 (反映频谱局部起伏剧烈程度)
    if len(mag) > 2:
        second_diff_energy = np.sum(np.diff(mag, n=2) ** 2)
    else:
        second_diff_energy = 0.0
    freq_extra[-1] = second_diff_energy

    # ---- STFT补充 (9) ----
    nperseg = min(256, N)
    f_stft, t_stft, Zxx = sp_signal.stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    stft_mag = np.abs(Zxx)
    stft_power = stft_mag ** 2
    frame_energy = np.sum(stft_power, axis=0)

    stft_energy_max = np.max(frame_energy)
    stft_energy_min = np.min(frame_energy)

    frame_power_sum = np.sum(stft_power, axis=0) + eps
    frame_centroids = np.sum(f_stft[:, None] * stft_power, axis=0) / frame_power_sum
    freq_centroid_max = np.max(frame_centroids)
    freq_centroid_min = np.min(frame_centroids)

    frame_spreads = np.sqrt(
        np.sum(((f_stft[:, None] - frame_centroids[None, :]) ** 2) * stft_power, axis=0) / frame_power_sum
    )
    stft_bw_mean = np.mean(frame_spreads)
    stft_bw_std = np.std(frame_spreads)
    stft_bw_max = np.max(frame_spreads)
    stft_bw_min = np.min(frame_spreads)

    if stft_mag.shape[1] > 1:
        flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
        flux_max = np.max(flux)
    else:
        flux_max = 0.0

    stft_extra = [stft_energy_max, stft_energy_min, freq_centroid_max, freq_centroid_min,
                  stft_bw_mean, stft_bw_std, stft_bw_max, stft_bw_min, flux_max]

    # ---- 包络/瞬时频率补充 (8) ----
    envelope_rms = np.sqrt(np.mean(envelope ** 2))
    envelope_skew = skew(envelope)
    envelope_crest = np.max(envelope) / (envelope_rms + eps)

    inst_freq_mean = np.mean(inst_freq)
    inst_freq_skew = skew(inst_freq)
    inst_freq_kurt = kurtosis(inst_freq)

    Env_X = fft(envelope - np.mean(envelope))
    half = N // 2
    env_mag = np.abs(Env_X[:half])
    env_power = env_mag ** 2
    env_power_sum = np.sum(env_power) + eps
    env_psd_norm = env_power / env_power_sum
    envelope_spectral_entropy = -np.sum(env_psd_norm * np.log2(env_psd_norm + eps)) / np.log2(len(env_psd_norm) + eps)
    envelope_spectral_flatness = np.exp(np.mean(np.log(env_power + eps))) / (np.mean(env_power) + eps)

    envelope_extra = [envelope_rms, envelope_skew, envelope_crest,
                       inst_freq_mean, inst_freq_skew, inst_freq_kurt,
                       envelope_spectral_entropy, envelope_spectral_flatness]

    extra_features = time_extra + freq_extra + stft_extra + envelope_extra
    return extra_features


FEATURE_NAMES_64_EXTRA = [
    "Median", "Variance", "Mean Absolute Value", "Median Absolute Deviation",
    "Shape Factor", "Impulse Factor",
    "Spectral Variance", "Spectral Crest Factor", "Spectral Decrease",
    "Roll-off 50%", "-6dB Bandwidth", "Spectral Energy", "Spectral Energy Ratio",
    "Spectral Peak-to-Average Ratio", "Spectral 2nd-Diff Energy",
    "STFT Energy Max", "STFT Energy Min", "Frequency Centroid Max", "Frequency Centroid Min",
    "STFT Bandwidth Mean", "STFT Bandwidth Std", "STFT Bandwidth Max", "STFT Bandwidth Min",
    "Spectral Flux Max",
    "Envelope RMS", "Envelope Skewness", "Envelope Crest Factor",
    "Instantaneous Frequency Mean", "Instantaneous Frequency Skewness",
    "Instantaneous Frequency Kurtosis",
    "Envelope Spectral Entropy", "Envelope Spectral Flatness"
]


# ---------------------------------------------------------
# 2.3 128维增量特征 (在64维基础上新增64个 —— 本次重点扩充)
# ---------------------------------------------------------
def extract_128_extra_features(x, fs, cached):
    """
    在64维特征基础上追加64个"高价值"特征，专注提升无人机与WiFi/蓝牙的区分度:
      A. 高阶累积量补充 (6):  C60/C63等高阶调制识别特征
      B. 多尺度小波能量特征 (16): 捕捉不同时间尺度的能量分布(db4小波4层分解)
      C. 分频带能量占比特征 (16): 将频谱等分为8个子频带，计算能量占比+能量方差
      D. 自相关特征 (8): 捕捉信号周期性/随机性 (无人机跳频 vs WiFi/BT固定信道)
      E. 瞬时幅度/相位调制域特征 (10): AM/PM调制指纹，常用于无线电指纹识别
      F. 信息熵类特征补充 (8): 排列熵、样本熵等非线性动力学特征
    """
    X, freqs, mag, power, analytic, envelope, phase, inst_freq = cached
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    eps = 1e-12

    # ================= A. 高阶累积量 (6) =================
    z = analytic
    M20 = np.mean(z ** 2)
    M21 = np.mean(np.abs(z) ** 2)
    M40 = np.mean(z ** 4)
    M42 = np.mean(np.abs(z) ** 4)
    M60 = np.mean(z ** 6)
    M63 = np.mean((z ** 3) * np.conj(z) ** 3)
    M61 = np.mean((z ** 5) * np.conj(z))

    C40 = M40 - 3 * (M20 ** 2)
    C60 = M60 - 15 * M20 * M40 + 30 * (M20 ** 3)
    C61 = M61 - 5 * M20 * M21 - 10 * np.conj(M20) * M40 * 0  # 简化项，工程近似
    C63 = np.abs(M63 - 9 * M21 * M42 + 12 * (M21 ** 3) - 3 * np.abs(M20) ** 2 * M63 * 0)

    # 归一化累积量(除以功率的对应次幂，得到尺度不变特征)
    C40_norm = np.abs(C40) / (M21 ** 2 + eps)
    C60_norm = np.abs(C60) / (M21 ** 3 + eps)
    C63_norm = np.abs(C63) / (M21 ** 3 + eps)

    cumulant_feats = [np.abs(C40), np.abs(C60), C40_norm, C60_norm, C63_norm, np.abs(M42)]

    # ================= B. 多尺度小波能量特征 (16) =================
    # db4小波4层分解，得到4个细节系数(cD1~cD4) + 1个近似系数(cA4)，共5个子带
    # 对每个子带计算: 能量占比 + 能量标准差，共5*2=10... 补齐到16个用更细的分解层数
    wavelet_feats = []
    try:
        coeffs = pywt.wavedec(x, 'db4', level=5)  # [cA5, cD5, cD4, cD3, cD2, cD1]
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = np.sum(energies) + eps
        energy_ratios = [e / total_energy for e in energies]   # 6个子带能量占比
        energy_stds = [np.std(c) for c in coeffs]               # 6个子带系数标准差
        energy_entropy = -np.sum([r * np.log2(r + eps) for r in energy_ratios])  # 小波能量熵(1个)

        # 补齐到16维: 6(占比) + 6(标准差) + 1(熵) + 相邻子带能量比值(3个,反映能量随尺度变化趋势)
        adjacent_ratios = [energies[i] / (energies[i+1] + eps) for i in range(min(3, len(energies)-1))]
        while len(adjacent_ratios) < 3:
            adjacent_ratios.append(0.0)

        wavelet_feats = energy_ratios + energy_stds + [energy_entropy] + adjacent_ratios
    except Exception:
        wavelet_feats = [0.0] * 16

    # 强制截断/补齐到16维，保证特征维度稳定
    wavelet_feats = (wavelet_feats + [0.0] * 16)[:16]

    # ================= C. 分频带能量占比特征 (16) =================
    n_bands = 8
    half = len(mag)
    band_edges = np.linspace(0, half, n_bands + 1).astype(int)
    power_sum_total = np.sum(power) + eps
    band_energy_ratios = []
    band_energy_stds = []
    for i in range(n_bands):
        band_power = power[band_edges[i]:band_edges[i+1]]
        band_energy_ratios.append(np.sum(band_power) / power_sum_total)
        band_energy_stds.append(np.std(band_power) if len(band_power) > 0 else 0.0)

    band_feats = band_energy_ratios + band_energy_stds  # 8+8=16

    # ================= D. 自相关特征 (8) =================
    x_centered = x - np.mean(x)
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]   # 只取非负延迟部分
    autocorr_norm = autocorr / (autocorr[0] + eps)   # 归一化，autocorr_norm[0]=1

    # 找到第一个过零点位置(反映信号周期性尺度)
    zero_cross_idx = np.where(np.diff(np.sign(autocorr_norm)))[0]
    first_zero_lag = zero_cross_idx[0] if len(zero_cross_idx) > 0 else len(autocorr_norm)

    # 自相关衰减特征: 取几个固定延迟点的自相关值 + 统计量
    lag_points = [1, 2, 4, 8]
    lag_vals = [autocorr_norm[lag] if lag < len(autocorr_norm) else 0.0 for lag in lag_points]

    autocorr_mean = np.mean(autocorr_norm[1:min(50, len(autocorr_norm))])
    autocorr_std = np.std(autocorr_norm[1:min(50, len(autocorr_norm))])
    autocorr_max_sidelobe = np.max(np.abs(autocorr_norm[1:min(50, len(autocorr_norm))])) if len(autocorr_norm) > 1 else 0.0

    autocorr_feats = lag_vals + [first_zero_lag, autocorr_mean, autocorr_std, autocorr_max_sidelobe]  # 4+4=8

    # ================= E. 瞬时幅度/相位调制域特征 (10) =================
    inst_amp = envelope  # 瞬时幅度就是包络
    inst_amp_norm = inst_amp / (np.mean(inst_amp) + eps)   # 归一化瞬时幅度(去除功率影响)

    # 瞬时幅度的高阶统计量(AM调制指纹)
    ia_std = np.std(inst_amp_norm)
    ia_kurt = kurtosis(inst_amp_norm)
    ia_skew = skew(inst_amp_norm)

    # 瞬时相位的非线性特征(PM调制指纹): 相位差分的统计量
    phase_diff = np.diff(phase)
    phase_diff_std = np.std(phase_diff)
    phase_diff_kurt = kurtosis(phase_diff)

    # 瞬时频率的二阶差分(反映频率变化的加速度，跳频信号该值会突变)
    if len(inst_freq) > 1:
        inst_freq_diff = np.diff(inst_freq)
        freq_accel_std = np.std(inst_freq_diff)
        freq_accel_max = np.max(np.abs(inst_freq_diff))
    else:
        freq_accel_std = 0.0
        freq_accel_max = 0.0

    # 瞬时幅度谱的峰度(反映幅度调制的规律性)
    Env_X = fft(inst_amp_norm - np.mean(inst_amp_norm))
    env_mag_norm = np.abs(Env_X[:len(Env_X)//2])
    env_spec_kurt = kurtosis(env_mag_norm) if len(env_mag_norm) > 3 else 0.0
    env_spec_peak_ratio = np.max(env_mag_norm) / (np.mean(env_mag_norm) + eps)

    modulation_feats = [ia_std, ia_kurt, ia_skew, phase_diff_std, phase_diff_kurt,
                          freq_accel_std, freq_accel_max, env_spec_kurt, env_spec_peak_ratio, 0.0]  # 补齐到10

    # ================= F. 信息熵类特征补充 (8) =================
    def permutation_entropy(signal, order=3, delay=1):
        """排列熵: 衡量时间序列的复杂度/随机性"""
        n = len(signal)
        if n < order * delay:
            return 0.0
        permutations = {}
        for i in range(n - (order - 1) * delay):
            pattern = tuple(np.argsort(signal[i:i + order * delay:delay]))
            permutations[pattern] = permutations.get(pattern, 0) + 1
        counts = np.array(list(permutations.values()))
        probs = counts / np.sum(counts)
        pe = -np.sum(probs * np.log2(probs + eps))
        max_pe = np.log2(np.math.factorial(order))
        return pe / (max_pe + eps)   # 归一化到[0,1]

    def sample_entropy(signal, m=2, r_ratio=0.2):
        """样本熵简化实现(采样降低计算量，避免O(N^2)在1024点上过慢)"""
        signal = signal[::2]   # 降采样加速计算
        N_s = len(signal)
        r = r_ratio * np.std(signal)
        if N_s < m + 2:
            return 0.0

        def _phi(m_val):
            templates = np.array([signal[i:i + m_val] for i in range(N_s - m_val + 1)])
            count = 0
            total = 0
            for i in range(len(templates)):
                dist = np.max(np.abs(templates - templates[i]), axis=1)
                count += np.sum(dist <= r) - 1   # 排除自身匹配
                total += len(templates) - 1
            return count / (total + eps)

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        return -np.log(phi_m1 / phi_m + eps)

    perm_entropy = permutation_entropy(x, order=3, delay=1)
    samp_entropy = sample_entropy(x, m=2, r_ratio=0.2)

    # 频谱的排列熵(反映频谱形状的复杂度)
    perm_entropy_spec = permutation_entropy(mag, order=3, delay=1)

    # 香农熵(基于信号幅值直方图)
    hist, _ = np.histogram(x, bins=32, density=True)
    hist_norm = hist / (np.sum(hist) + eps)
    shannon_entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + eps))

    # 近似熵简化版(用与样本熵相同框架，r_ratio不同)
    approx_entropy_proxy = sample_entropy(x, m=2, r_ratio=0.15)

    entropy_feats = [perm_entropy, samp_entropy, perm_entropy_spec, shannon_entropy,
                      approx_entropy_proxy, 0.0, 0.0, 0.0]   # 补齐到8

    extra_128_features = (cumulant_feats + wavelet_feats + band_feats +
                           autocorr_feats + modulation_feats + entropy_feats)

    # 确保总数恰好64
    extra_128_features = (extra_128_features + [0.0] * 64)[:64]
    return extra_128_features


FEATURE_NAMES_128_EXTRA = (
    ["|C40|", "|C60|", "C40_norm", "C60_norm", "C63_norm", "|M42|"] +                      # 6
    [f"Wavelet_EnergyRatio_{i}" for i in range(6)] +
    [f"Wavelet_Std_{i}" for i in range(6)] +
    ["Wavelet_Entropy"] +
    [f"Wavelet_AdjRatio_{i}" for i in range(3)] +                                          # 16
    [f"Band_EnergyRatio_{i}" for i in range(8)] +
    [f"Band_EnergyStd_{i}" for i in range(8)] +                                            # 16
    [f"Autocorr_Lag_{lag}" for lag in [1, 2, 4, 8]] +
    ["Autocorr_FirstZeroLag", "Autocorr_Mean", "Autocorr_Std", "Autocorr_MaxSidelobe"] +   # 8
    ["InstAmp_Std", "InstAmp_Kurt", "InstAmp_Skew", "PhaseDiff_Std", "PhaseDiff_Kurt",
     "FreqAccel_Std", "FreqAccel_Max", "EnvSpec_Kurt", "EnvSpec_PeakRatio", "Modulation_Reserved"] +  # 10
    ["PermEntropy", "SampleEntropy", "PermEntropy_Spectrum", "ShannonEntropy",
     "ApproxEntropy_Proxy", "Entropy_Reserved1", "Entropy_Reserved2", "Entropy_Reserved3"]  # 8
)
assert len(FEATURE_NAMES_128_EXTRA) == 64


# ---------------------------------------------------------
# 2.4 统一特征提取接口
# ---------------------------------------------------------
def get_feature_names(feature_dim):
    if feature_dim == 32:
        return FEATURE_NAMES_32
    elif feature_dim == 64:
        return FEATURE_NAMES_32 + FEATURE_NAMES_64_EXTRA
    elif feature_dim == 128:
        return FEATURE_NAMES_32 + FEATURE_NAMES_64_EXTRA + FEATURE_NAMES_128_EXTRA
    else:
        raise ValueError(f"不支持的特征维度: {feature_dim}，仅支持32/64/128")


def extract_features_single(x, fs=1.0, feature_dim=32):
    """
    统一特征提取入口，根据 feature_dim 返回对应维度的特征向量
    """
    base_feats, cached = extract_32_features(x, fs=fs)

    if feature_dim == 32:
        all_feats = base_feats
    elif feature_dim == 64:
        extra_64 = extract_64_extra_features(x, fs, cached)
        all_feats = base_feats + extra_64
    elif feature_dim == 128:
        extra_64 = extract_64_extra_features(x, fs, cached)
        extra_128 = extract_128_extra_features(x, fs, cached)
        all_feats = base_feats + extra_64 + extra_128
    else:
        raise ValueError(f"不支持的特征维度: {feature_dim}，仅支持32/64/128")

    all_feats = np.array(all_feats, dtype=np.float64)
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(all_feats) == feature_dim, f"特征维度不匹配: 期望{feature_dim}, 实际{len(all_feats)}"
    return all_feats


def extract_features_batch(X_raw, fs=1.0, feature_dim=32, tag=""):
    feats = np.zeros((X_raw.shape[0], feature_dim), dtype=np.float64)
    for i in range(X_raw.shape[0]):
        feats[i] = extract_features_single(X_raw[i], fs=fs, feature_dim=feature_dim)
        if (i + 1) % 5000 == 0:
            print(f"  [{tag}] 特征提取进度: {i+1}/{X_raw.shape[0]}")
    return feats


# =========================================================
# 3. LOF 检测器 (纯LOF，不再融合KNN)
# =========================================================
class LOFDetector:
    """
    单独 LOF 检测器
    统一接口: fit(X_train) / score(X) [越大越异常] / predict(X) [1=异常, 0=正常]
    阈值完全由 pyod 内部基于训练集分数分布 + contamination 参数自动确定，
    不使用任何无人机数据
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.clf = LOF(
            n_neighbors=cfg.LOF_N_NEIGHBORS,
            metric=cfg.LOF_METRIC,
            contamination=cfg.CONTAMINATION,
        )

    def fit(self, X_train):
        self.clf.fit(X_train)
        return self

    def score(self, X):
        return self.clf.decision_function(X)

    def predict(self, X):
        return self.clf.predict(X)


# =========================================================
# 4. 评估
# =========================================================
def evaluate(y_true_binary, pred01, scores, title=""):
    y_true01 = (y_true_binary == -1).astype(int)

    acc = accuracy_score(y_true01, pred01)
    prec = precision_score(y_true01, pred01, zero_division=0)
    rec = recall_score(y_true01, pred01, zero_division=0)
    f1 = f1_score(y_true01, pred01, zero_division=0)
    try:
        auc = roc_auc_score(y_true01, scores)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true01, pred01)

    print("=" * 55)
    if title:
        print(f"【{title}】")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("混淆矩阵 [0=正常干扰, 1=无人机异常] (行=真实, 列=预测)")
    print(cm)
    print("=" * 55)

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "confusion_matrix": cm}


def analyze_per_drone_model(detector, feat_drone, y_drone_labels):
    scores = detector.score(feat_drone)
    pred01 = detector.predict(feat_drone)

    df = pd.DataFrame({
        "drone_model": y_drone_labels,
        "score": scores,
        "detected": pred01
    })

    summary = df.groupby("drone_model").agg(
        total_samples=("detected", "count"),
        detected_count=("detected", "sum"),
        recall=("detected", "mean"),
        mean_score=("score", "mean"),
    ).sort_values("recall")

    print("\n" + "=" * 60)
    print("各无人机型号检测召回率明细 (按recall升序):")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)

    return summary


# =========================================================
# 5. 主流程
# =========================================================
def main():
    cfg = Config()
    set_seed(cfg.RANDOM_STATE)

    print(f"当前特征维度: {cfg.FEATURE_DIM}")

    # ---------------- 数据加载 ----------------
    print("\n加载 WiFi 数据 ...")
    X_wifi, _ = load_dataset(cfg.DATA_ROOT, cfg.WIFI_DIRS)
    print(f"  WiFi 样本数: {X_wifi.shape[0]}")

    print("加载 蓝牙 数据 ...")
    X_bt, _ = load_dataset(cfg.DATA_ROOT, cfg.BT_DIRS)
    print(f"  蓝牙样本数: {X_bt.shape[0]}")

    print("加载 无人机 数据 ...")
    X_drone, y_drone_model = load_dataset(cfg.DATA_ROOT, cfg.DRONE_DIRS)
    print(f"  无人机样本数: {X_drone.shape[0]}")

    X_normal = np.concatenate([X_wifi, X_bt], axis=0)

    # ---------------- 数据划分 ----------------
    X_norm_train, X_norm_test = train_test_split(
        X_normal, test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE, shuffle=True
    )
    X_drone_test, y_drone_test = X_drone, y_drone_model

    print(f"\n数据划分结果:")
    print(f"  正常-训练集: {X_norm_train.shape[0]}")
    print(f"  正常-测试集: {X_norm_test.shape[0]}")
    print(f"  无人机-测试集(全部,不参与训练/调参): {X_drone_test.shape[0]}")

    # ---------------- 特征提取 ----------------
    print(f"\n提取训练集特征 (维度={cfg.FEATURE_DIM}) ...")
    feat_norm_train_raw = extract_features_batch(
        X_norm_train, fs=cfg.FS, feature_dim=cfg.FEATURE_DIM, tag="训练集")

    print("提取正常测试集特征 ...")
    feat_norm_test_raw = extract_features_batch(
        X_norm_test, fs=cfg.FS, feature_dim=cfg.FEATURE_DIM, tag="正常测试集")

    print("提取无人机测试集特征 ...")
    feat_drone_test_raw = extract_features_batch(
        X_drone_test, fs=cfg.FS, feature_dim=cfg.FEATURE_DIM, tag="无人机测试集")

    # ---------------- 标准化 ----------------
    scaler = StandardScaler()
    feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
    feat_norm_test = scaler.transform(feat_norm_test_raw)
    feat_drone_test = scaler.transform(feat_drone_test_raw)

    # ---------------- 训练LOF ----------------
    print(f"\n训练 LOF 检测器 (特征维度={cfg.FEATURE_DIM}) ...")
    detector = LOFDetector(cfg)
    detector.fit(feat_norm_train)

    # ---------------- 最终测试评估 ----------------
    feat_test_all = np.concatenate([feat_norm_test, feat_drone_test], axis=0)
    y_test_binary = np.concatenate([
        np.ones(feat_norm_test.shape[0]),
        -np.ones(feat_drone_test.shape[0]),
    ])

    scores_test = detector.score(feat_test_all)
    pred_test = detector.predict(feat_test_all)

    metrics = evaluate(y_test_binary, pred_test, scores_test,
                        title=f"最终测试结果 - LOF (特征维度={cfg.FEATURE_DIM})")

    analyze_per_drone_model(detector, feat_drone_test, y_drone_test)

    # ---------------- 保存模型 ----------------
    joblib.dump({
        "detector": detector,
        "scaler": scaler,
        "feature_dim": cfg.FEATURE_DIM,
        "feature_names": get_feature_names(cfg.FEATURE_DIM)
    }, f"lof_detector_dim{cfg.FEATURE_DIM}.pkl")
    print(f"\n模型已保存至: lof_detector_dim{cfg.FEATURE_DIM}.pkl")

    return detector, scaler, metrics


# =========================================================
# 6. 三档特征维度对比运行 (可选)
# =========================================================
def run_all_dims_and_compare():
    """依次运行 32/64/128 三档特征维度，输出对比表格"""
    cfg_base = Config()
    set_seed(cfg_base.RANDOM_STATE)

    print("加载数据 ...")
    X_wifi, _ = load_dataset(cfg_base.DATA_ROOT, cfg_base.WIFI_DIRS)
    X_bt, _ = load_dataset(cfg_base.DATA_ROOT, cfg_base.BT_DIRS)
    X_drone, y_drone_model = load_dataset(cfg_base.DATA_ROOT, cfg_base.DRONE_DIRS)
    X_normal = np.concatenate([X_wifi, X_bt], axis=0)

    X_norm_train, X_norm_test = train_test_split(
        X_normal, test_size=cfg_base.TEST_SIZE,
        random_state=cfg_base.RANDOM_STATE, shuffle=True
    )
    X_drone_test, y_drone_test = X_drone, y_drone_model

    all_metrics = {}
    for dim in [32, 64, 128]:
        print("\n" + "#" * 70)
        print(f"特征维度: {dim}")
        print("#" * 70)

        cfg = Config()
        cfg.FEATURE_DIM = dim

        feat_norm_train_raw = extract_features_batch(X_norm_train, fs=cfg.FS, feature_dim=dim, tag=f"训练集-dim{dim}")
        feat_norm_test_raw = extract_features_batch(X_norm_test, fs=cfg.FS, feature_dim=dim, tag=f"正常测试-dim{dim}")
        feat_drone_test_raw = extract_features_batch(X_drone_test, fs=cfg.FS, feature_dim=dim, tag=f"无人机测试-dim{dim}")

        scaler = StandardScaler()
        feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
        feat_norm_test = scaler.transform(feat_norm_test_raw)
        feat_drone_test = scaler.transform(feat_drone_test_raw)

        detector = LOFDetector(cfg)
        detector.fit(feat_norm_train)

        feat_test_all = np.concatenate([feat_norm_test, feat_drone_test], axis=0)
        y_test_binary = np.concatenate([
            np.ones(feat_norm_test.shape[0]), -np.ones(feat_drone_test.shape[0])
        ])

        scores_test = detector.score(feat_test_all)
        pred_test = detector.predict(feat_test_all)
        metrics = evaluate(y_test_binary, pred_test, scores_test, title=f"LOF-dim{dim}")

        analyze_per_drone_model(detector, feat_drone_test, y_drone_test)

        all_metrics[f"LOF-dim{dim}"] = metrics

    print("\n" + "=" * 70)
    print("三档特征维度对比汇总:")
    print("=" * 70)
    df_compare = pd.DataFrame(all_metrics).T[["accuracy", "precision", "recall", "f1", "auc"]]
    print(df_compare.to_string())
    df_compare.to_csv("feature_dim_comparison.csv")

    return df_compare


if __name__ == "__main__":
    # 方式1: 只运行 Config 中当前配置的单一维度
    main()

    # 方式2: 若想一次性对比32/64/128三档，注释掉上面 main()，改用下面这行
    # run_all_dims_and_compare()