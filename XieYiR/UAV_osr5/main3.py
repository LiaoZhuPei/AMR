"""
Feature-LOF 无人机射频信号异常检测（64维手工特征版本）
========================================================
第一级：64维特征提取 + LOF(局部离群因子) 做异常检测
第二级（可选扩展）：对被判定为无人机的样本进一步做型号分类
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
import joblib


# =========================================================
# 0. 全局配置
# =========================================================
class Config:
    DATA_ROOT = "./data_root"
    FS = 1.0                       # 采样率，若已知真实采样率请修改
    LOF_K = 20
    LOF_METRIC = "minkowski"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_FEATURES = 64

    WIFI_DIRS = ["TPLINK_TL_WR940N", "CISCO_LINKSYS_E3200"]
    BT_DIRS = ["APPLE_IPAD3", "APPLE_IPHONE6S", "APPLE_IPHONE7",
               "FITBIT_CHARGE3", "MOTOROLA"]
    DRONE_DIRS = ["DJI_M600", "BEEBEERUN", "DJI_INSPIRE",
                  "DJI_MAVICPRO", "DJI_PHANTOM"]

    FEATURE_NAMES = [
        # ---- 1~16 时域特征 ----
        "Mean", "Median", "Std", "Variance", "RMS", "Mean Absolute Value",
        "Maximum", "Minimum", "Peak-to-Peak", "Median Absolute Deviation",
        "Skewness", "Kurtosis", "Crest Factor", "Shape Factor",
        "Impulse Factor", "Zero Crossing Rate",
        # ---- 17~36 频域特征 ----
        "Peak Frequency", "Spectral Centroid", "Spectral Spread", "Spectral Variance",
        "Spectral Skewness", "Spectral Kurtosis", "Spectral Entropy", "Spectral Flatness",
        "Spectral Crest Factor", "Spectral Slope", "Spectral Decrease",
        "Roll-off 50%", "Roll-off 85%", "Roll-off 95%",
        "-3dB Bandwidth", "-6dB Bandwidth", "-10dB Bandwidth",
        "Spectral Energy", "Spectral Energy Ratio", "Spectral Peak-to-Average Ratio",
        # ---- 37~52 STFT时频特征 ----
        "STFT Energy Mean", "STFT Energy Std", "STFT Energy Max", "STFT Energy Min",
        "Frequency Centroid Mean", "Frequency Centroid Std",
        "Frequency Centroid Max", "Frequency Centroid Min",
        "STFT Bandwidth Mean", "STFT Bandwidth Std",
        "STFT Bandwidth Max", "STFT Bandwidth Min",
        "Spectral Flux Mean", "Spectral Flux Std", "Spectral Flux Max", "Spectral Flux Median",
        # ---- 53~64 包络与瞬时频率特征 ----
        "Envelope Mean", "Envelope Std", "Envelope RMS", "Envelope Skewness",
        "Envelope Kurtosis", "Envelope Crest Factor",
        "Instantaneous Frequency Mean", "Instantaneous Frequency Std",
        "Instantaneous Frequency Skewness", "Instantaneous Frequency Kurtosis",
        "Envelope Spectral Entropy", "Envelope Spectral Flatness"
    ]
    assert len(FEATURE_NAMES) == N_FEATURES


def set_seed(seed):
    np.random.seed(seed)


# =========================================================
# 1. 数据加载 (与之前一致，丢弃第1025列，文件夹名为标签)
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
# 2. 64维特征提取
# =========================================================
def extract_64_features(x, fs=1.0):
    """
    x: 1024维原始射频信号（1维numpy数组）
    返回: 64维特征向量，顺序严格对应 Config.FEATURE_NAMES
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    eps = 1e-12

    # =====================================================
    # 1~16 时域特征
    # =====================================================
    mean_val = np.mean(x)
    median_val = np.median(x)
    std_val = np.std(x)
    var_val = np.var(x)
    rms_val = np.sqrt(np.mean(x ** 2))
    mav_val = np.mean(np.abs(x))
    max_val = np.max(x)
    min_val = np.min(x)
    p2p = max_val - min_val
    mad_val = np.median(np.abs(x - median_val))          # Median Absolute Deviation
    skewness = skew(x)
    kurt = kurtosis(x)                                     # Fisher定义
    peak = np.max(np.abs(x))
    crest_factor = peak / (rms_val + eps)
    shape_factor = rms_val / (mav_val + eps)
    impulse_factor = peak / (mav_val + eps)
    zc = np.sum(np.abs(np.diff(np.sign(x))) > 0)
    zcr = zc / (N - 1)

    time_feats = [mean_val, median_val, std_val, var_val, rms_val, mav_val,
                  max_val, min_val, p2p, mad_val, skewness, kurt,
                  crest_factor, shape_factor, impulse_factor, zcr]

    # =====================================================
    # 17~36 频域特征
    # =====================================================
    X = fft(x)
    half = N // 2
    mag = np.abs(X[:half])
    freqs = fftfreq(N, d=1.0 / fs)[:half]
    power = mag ** 2
    power_sum = np.sum(power) + eps

    # 17 峰值频率
    peak_idx = np.argmax(mag)
    peak_freq = freqs[peak_idx]
    peak_mag = mag[peak_idx]

    # 18~22 频谱矩特征
    spectral_centroid = np.sum(freqs * power) / power_sum
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / power_sum)
    spectral_variance = spectral_spread ** 2
    spectral_skew = np.sum(((freqs - spectral_centroid) ** 3) * power) / (
            power_sum * (spectral_spread ** 3 + eps))
    spectral_kurt = np.sum(((freqs - spectral_centroid) ** 4) * power) / (
            power_sum * (spectral_spread ** 4 + eps)) - 3

    # 23 频谱熵
    psd_norm = power / power_sum
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + eps)) / np.log2(len(psd_norm) + eps)

    # 24 频谱平坦度
    spectral_flatness = np.exp(np.mean(np.log(power + eps))) / (np.mean(power) + eps)

    # 25 频谱峰值因子 (幅度域 crest-like)
    spectral_crest_factor = peak_mag / (np.mean(mag) + eps)

    # 26 频谱斜率 (log幅度对频率线性拟合)
    log_mag = np.log(mag + eps)
    slope, _ = np.polyfit(freqs, log_mag, 1)
    spectral_slope = slope

    # 27 频谱衰减 (Spectral Decrease)
    k_idx = np.arange(1, len(mag))
    spectral_decrease = np.sum((mag[1:] - mag[0]) / k_idx) / (np.sum(mag[1:]) + eps)

    # 28~30 滚降点
    cumsum_power = np.cumsum(power)

    def rolloff(ratio):
        idx = np.searchsorted(cumsum_power, ratio * power_sum)
        idx = min(idx, len(freqs) - 1)
        return freqs[idx]

    rolloff_50 = rolloff(0.50)
    rolloff_85 = rolloff(0.85)
    rolloff_95 = rolloff(0.95)

    # 31~33 -3dB / -6dB / -10dB 带宽
    def bandwidth_at_db(mag, freqs, peak_mag, db):
        threshold = peak_mag / (10 ** (db / 20.0))
        idx = np.where(mag >= threshold)[0]
        if len(idx) == 0:
            return 0.0
        return freqs[idx[-1]] - freqs[idx[0]]

    bw_3db = bandwidth_at_db(mag, freqs, peak_mag, 3)
    bw_6db = bandwidth_at_db(mag, freqs, peak_mag, 6)
    bw_10db = bandwidth_at_db(mag, freqs, peak_mag, 10)

    # 34 频谱能量
    spectral_energy = power_sum

    # 35 频谱能量占比 (取 -3dB 带宽范围内能量 / 总能量)
    idx_band = np.where(mag >= peak_mag / (10 ** (3 / 20.0)))[0]
    if len(idx_band) > 0:
        band_energy = np.sum(power[idx_band])
    else:
        band_energy = 0.0
    spectral_energy_ratio = band_energy / power_sum

    # 36 频谱峰均比 (Peak-to-Average Power Ratio)
    spectral_papr = peak_mag ** 2 / (np.mean(power) + eps)

    freq_feats = [peak_freq, spectral_centroid, spectral_spread, spectral_variance,
                  spectral_skew, spectral_kurt, spectral_entropy, spectral_flatness,
                  spectral_crest_factor, spectral_slope, spectral_decrease,
                  rolloff_50, rolloff_85, rolloff_95,
                  bw_3db, bw_6db, bw_10db,
                  spectral_energy, spectral_energy_ratio, spectral_papr]

    # =====================================================
    # 37~52 STFT 时频特征
    # =====================================================
    nperseg = min(256, N)
    f_stft, t_stft, Zxx = sp_signal.stft(x, fs=fs, nperseg=nperseg,
                                          noverlap=nperseg // 2)
    stft_mag = np.abs(Zxx)                                  # (freq_bins, n_frames)
    stft_power = stft_mag ** 2

    # 37~40 STFT能量统计 (每帧总能量)
    frame_energy = np.sum(stft_power, axis=0)
    stft_energy_mean = np.mean(frame_energy)
    stft_energy_std = np.std(frame_energy)
    stft_energy_max = np.max(frame_energy)
    stft_energy_min = np.min(frame_energy)

    # 41~44 每帧频率质心统计
    frame_power_sum = np.sum(stft_power, axis=0) + eps
    frame_centroids = np.sum(f_stft[:, None] * stft_power, axis=0) / frame_power_sum
    freq_centroid_mean = np.mean(frame_centroids)
    freq_centroid_std = np.std(frame_centroids)
    freq_centroid_max = np.max(frame_centroids)
    freq_centroid_min = np.min(frame_centroids)

    # 45~48 每帧频谱带宽(spread)统计
    frame_spreads = np.sqrt(
        np.sum(((f_stft[:, None] - frame_centroids[None, :]) ** 2) * stft_power, axis=0)
        / frame_power_sum
    )
    stft_bw_mean = np.mean(frame_spreads)
    stft_bw_std = np.std(frame_spreads)
    stft_bw_max = np.max(frame_spreads)
    stft_bw_min = np.min(frame_spreads)

    # 49~52 谱通量统计 (相邻帧幅度差的欧氏范数)
    if stft_mag.shape[1] > 1:
        flux = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
        flux_mean = np.mean(flux)
        flux_std = np.std(flux)
        flux_max = np.max(flux)
        flux_median = np.median(flux)
    else:
        flux_mean = flux_std = flux_max = flux_median = 0.0

    stft_feats = [stft_energy_mean, stft_energy_std, stft_energy_max, stft_energy_min,
                  freq_centroid_mean, freq_centroid_std, freq_centroid_max, freq_centroid_min,
                  stft_bw_mean, stft_bw_std, stft_bw_max, stft_bw_min,
                  flux_mean, flux_std, flux_max, flux_median]

    # =====================================================
    # 53~64 包络 & 瞬时频率特征
    # =====================================================
    analytic = sp_signal.hilbert(x)
    envelope = np.abs(analytic)

    envelope_mean = np.mean(envelope)
    envelope_std = np.std(envelope)
    envelope_rms = np.sqrt(np.mean(envelope ** 2))
    envelope_skew = skew(envelope)
    envelope_kurt = kurtosis(envelope)
    envelope_crest = np.max(envelope) / (envelope_rms + eps)

    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    inst_freq_mean = np.mean(inst_freq)
    inst_freq_std = np.std(inst_freq)
    inst_freq_skew = skew(inst_freq)
    inst_freq_kurt = kurtosis(inst_freq)

    # 63~64 包络谱熵 / 平坦度 (对包络信号做FFT后计算)
    Env_X = fft(envelope - np.mean(envelope))
    env_mag = np.abs(Env_X[:half])
    env_power = env_mag ** 2
    env_power_sum = np.sum(env_power) + eps
    env_psd_norm = env_power / env_power_sum
    envelope_spectral_entropy = -np.sum(env_psd_norm * np.log2(env_psd_norm + eps)) / np.log2(len(env_psd_norm) + eps)
    envelope_spectral_flatness = np.exp(np.mean(np.log(env_power + eps))) / (np.mean(env_power) + eps)

    envelope_feats = [envelope_mean, envelope_std, envelope_rms, envelope_skew,
                       envelope_kurt, envelope_crest,
                       inst_freq_mean, inst_freq_std, inst_freq_skew, inst_freq_kurt,
                       envelope_spectral_entropy, envelope_spectral_flatness]

    # =====================================================
    # 拼接全部特征
    # =====================================================
    features = np.array(time_feats + freq_feats + stft_feats + envelope_feats, dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    assert len(features) == 64, f"特征维度错误: {len(features)}"
    return features


def extract_features_batch(X_raw, fs=1.0, n_features=64):
    """对整批原始信号 (n_samples, 1024) 批量提取特征，返回 (n_samples, n_features)"""
    feats = np.zeros((X_raw.shape[0], n_features), dtype=np.float64)
    for i in range(X_raw.shape[0]):
        feats[i] = extract_64_features(X_raw[i], fs=fs)
        if (i + 1) % 2000 == 0:
            print(f"  特征提取进度: {i+1}/{X_raw.shape[0]}")
    return feats


# =========================================================
# 3. LOF 训练 / 推理
# =========================================================
def train_lof(feat_train, cfg):
    lof = LocalOutlierFactor(
        n_neighbors=cfg.LOF_K,
        metric=cfg.LOF_METRIC,
        novelty=True,
        contamination="auto",
    )
    lof.fit(feat_train)
    return lof


def detect_anomaly(lof, feat):
    pred = lof.predict(feat)               # 1=正常(inlier)，-1=异常(outlier)
    score = -lof.score_samples(feat)       # 值越大越可能异常
    return pred, score


# =========================================================
# 4. 评估
# =========================================================
def evaluate(y_true_binary, pred, scores):
    y_true01 = (y_true_binary == -1).astype(int)
    y_pred01 = (pred == -1).astype(int)

    acc = accuracy_score(y_true01, y_pred01)
    prec = precision_score(y_true01, y_pred01, zero_division=0)
    rec = recall_score(y_true01, y_pred01, zero_division=0)
    f1 = f1_score(y_true01, y_pred01, zero_division=0)
    try:
        auc = roc_auc_score(y_true01, scores)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true01, y_pred01)

    print("=" * 50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("混淆矩阵 [0=正常干扰, 1=无人机异常] (行=真实, 列=预测)")
    print(cm)
    print("=" * 50)

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "confusion_matrix": cm}


# =========================================================
# 5. (可选扩展) 第二级：无人机型号分类
# =========================================================
def train_drone_classifier(feat_drone, y_drone, cfg):
    h_train, h_test, y_train, y_test = train_test_split(
        feat_drone, y_drone, test_size=0.2,
        random_state=cfg.RANDOM_STATE, stratify=y_drone
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=cfg.RANDOM_STATE)
    clf.fit(h_train, y_train)
    acc = clf.score(h_test, y_test)
    print(f"[第二级-型号分类] 测试集准确率: {acc:.4f}")
    return clf


# =========================================================
# 6. 主流程
# =========================================================
def main():
    cfg = Config()
    set_seed(cfg.RANDOM_STATE)

    print("加载 WiFi 数据 ...")
    X_wifi, y_wifi = load_dataset(cfg.DATA_ROOT, cfg.WIFI_DIRS)
    print(f"  WiFi 样本数: {X_wifi.shape[0]}")

    print("加载 蓝牙 数据 ...")
    X_bt, y_bt = load_dataset(cfg.DATA_ROOT, cfg.BT_DIRS)
    print(f"  蓝牙样本数: {X_bt.shape[0]}")

    print("加载 无人机 数据 ...")
    X_drone, y_drone = load_dataset(cfg.DATA_ROOT, cfg.DRONE_DIRS)
    print(f"  无人机样本数: {X_drone.shape[0]}")

    X_normal = np.concatenate([X_wifi, X_bt], axis=0)

    X_norm_train, X_norm_test = train_test_split(
        X_normal, test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE, shuffle=True
    )
    print(f"正常训练样本: {X_norm_train.shape[0]}, 正常测试样本: {X_norm_test.shape[0]}")
    print(f"无人机(全部作为测试期异常样本): {X_drone.shape[0]}")

    # ---------- 1. 64维特征提取 ----------
    print("提取训练集特征 ...")
    feat_norm_train_raw = extract_features_batch(X_norm_train, fs=cfg.FS, n_features=cfg.N_FEATURES)
    print("提取正常测试集特征 ...")
    feat_norm_test_raw = extract_features_batch(X_norm_test, fs=cfg.FS, n_features=cfg.N_FEATURES)
    print("提取无人机测试集特征 ...")
    feat_drone_raw = extract_features_batch(X_drone, fs=cfg.FS, n_features=cfg.N_FEATURES)

    # ---------- 2. 标准化 (基于训练集拟合) ----------
    scaler = StandardScaler()
    feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
    feat_norm_test = scaler.transform(feat_norm_test_raw)
    feat_drone = scaler.transform(feat_drone_raw)

    # ---------- 3. 训练 LOF ----------
    lof = train_lof(feat_norm_train, cfg)

    # ---------- 4. 测试推理 ----------
    feat_test_all = np.concatenate([feat_norm_test, feat_drone], axis=0)
    y_test_binary = np.concatenate([
        np.ones(feat_norm_test.shape[0]),
        -np.ones(feat_drone.shape[0]),
    ])

    pred, scores = detect_anomaly(lof, feat_test_all)
    metrics = evaluate(y_test_binary, pred, scores)

    # ---------- 5. (可选) 第二级型号分类 ----------
    if feat_drone.shape[0] > 0:
        drone_clf = train_drone_classifier(feat_drone_raw, y_drone, cfg)
        joblib.dump(drone_clf, "drone_classifier.pkl")

    # ---------- 6. 保存模型 ----------
    joblib.dump({"lof": lof, "scaler": scaler,
                 "feature_names": cfg.FEATURE_NAMES}, "lof_feature_scaler.pkl")

    return lof, scaler, metrics


if __name__ == "__main__":
    main()