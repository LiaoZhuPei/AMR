"""
SDAE-LOF 项目 —— 纯64维高价值特征方案 (不含原始32维基础特征) + 纯LOF检测器
========================================================================
特征构成说明:
  仅保留64个"高价值"特征:
    A. 高阶累积量补充 (6)
    B. 多尺度小波能量特征 (16)
    C. 分频带能量占比特征 (16)
    D. 自相关特征 (8)
    E. 瞬时幅度/相位调制域特征 (10)
    F. 信息熵类特征 (8)
  完全不包含 Mean/Std/RMS/Skewness/Kurtosis/Spectral Centroid 等原始32维基础统计特征

数据流程:
  原始CSV -> 加载(丢弃第1025列, 文件夹名为标签) -> 64维高价值特征提取 -> 标准化
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

    TEST_SIZE = 0.2

    # ---- 本版本固定为64维 (纯高价值特征，不含原始32维基础特征) ----
    FEATURE_DIM = 64

    CONTAMINATION = 0.1             # 阈值确定完全基于训练集自身分布，不使用无人机数据

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
# 2. 中间量计算 (供高价值特征复用，但不作为最终特征输出)
# =========================================================
def compute_cached_intermediates(x, fs=1.0):
    """
    仅计算FFT/Hilbert等中间量，不生成任何基础统计特征
    返回: cached中间量元组，供 extract_64_highvalue_features 使用
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)

    X = fft(x)
    half = N // 2
    mag = np.abs(X[:half])
    freqs = fftfreq(N, d=1.0 / fs)[:half]
    power = mag ** 2

    analytic = sp_signal.hilbert(x)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs

    cached = (X, freqs, mag, power, analytic, envelope, phase, inst_freq)
    return cached


# =========================================================
# 3. 64维高价值特征 (与之前版本一致，独立作为唯一特征来源)
# =========================================================
def extract_64_highvalue_features(x, fs, cached):
    """
    64个高价值特征，专注提升无人机与WiFi/蓝牙的区分度:
      A. 高阶累积量补充 (6):  C60/C63等高阶调制识别特征
      B. 多尺度小波能量特征 (16): 捕捉不同时间尺度的能量分布(db4小波5层分解)
      C. 分频带能量占比特征 (16): 将频谱等分为8个子频带，计算能量占比+能量方差
      D. 自相关特征 (8): 捕捉信号周期性/随机性 (无人机跳频 vs WiFi/BT固定信道)
      E. 瞬时幅度/相位调制域特征 (10): AM/PM调制指纹，常用于无线电指纹识别
      F. 信息熵类特征 (8): 排列熵、样本熵等非线性动力学特征
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

    C40 = M40 - 3 * (M20 ** 2)
    C60 = M60 - 15 * M20 * M40 + 30 * (M20 ** 3)
    C63 = np.abs(M63 - 9 * M21 * M42 + 12 * (M21 ** 3))

    C40_norm = np.abs(C40) / (M21 ** 2 + eps)
    C60_norm = np.abs(C60) / (M21 ** 3 + eps)
    C63_norm = C63 / (M21 ** 3 + eps)

    cumulant_feats = [np.abs(C40), np.abs(C60), C40_norm, C60_norm, C63_norm, np.abs(M42)]

    # ================= B. 多尺度小波能量特征 (16) =================
    wavelet_feats = []
    try:
        coeffs = pywt.wavedec(x, 'db4', level=5)  # [cA5, cD5, cD4, cD3, cD2, cD1]
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = np.sum(energies) + eps
        energy_ratios = [e / total_energy for e in energies]
        energy_stds = [np.std(c) for c in coeffs]
        energy_entropy = -np.sum([r * np.log2(r + eps) for r in energy_ratios])

        adjacent_ratios = [energies[i] / (energies[i+1] + eps) for i in range(min(3, len(energies)-1))]
        while len(adjacent_ratios) < 3:
            adjacent_ratios.append(0.0)

        wavelet_feats = energy_ratios + energy_stds + [energy_entropy] + adjacent_ratios
    except Exception:
        wavelet_feats = [0.0] * 16

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
    autocorr = autocorr[len(autocorr)//2:]
    autocorr_norm = autocorr / (autocorr[0] + eps)

    zero_cross_idx = np.where(np.diff(np.sign(autocorr_norm)))[0]
    first_zero_lag = zero_cross_idx[0] if len(zero_cross_idx) > 0 else len(autocorr_norm)

    lag_points = [1, 2, 4, 8]
    lag_vals = [autocorr_norm[lag] if lag < len(autocorr_norm) else 0.0 for lag in lag_points]

    autocorr_mean = np.mean(autocorr_norm[1:min(50, len(autocorr_norm))])
    autocorr_std = np.std(autocorr_norm[1:min(50, len(autocorr_norm))])
    autocorr_max_sidelobe = np.max(np.abs(autocorr_norm[1:min(50, len(autocorr_norm))])) if len(autocorr_norm) > 1 else 0.0

    autocorr_feats = lag_vals + [first_zero_lag, autocorr_mean, autocorr_std, autocorr_max_sidelobe]

    # ================= E. 瞬时幅度/相位调制域特征 (10) =================
    inst_amp = envelope
    inst_amp_norm = inst_amp / (np.mean(inst_amp) + eps)

    ia_std = np.std(inst_amp_norm)
    ia_kurt = kurtosis(inst_amp_norm)
    ia_skew = skew(inst_amp_norm)

    phase_diff = np.diff(phase)
    phase_diff_std = np.std(phase_diff)
    phase_diff_kurt = kurtosis(phase_diff)

    if len(inst_freq) > 1:
        inst_freq_diff = np.diff(inst_freq)
        freq_accel_std = np.std(inst_freq_diff)
        freq_accel_max = np.max(np.abs(inst_freq_diff))
    else:
        freq_accel_std = 0.0
        freq_accel_max = 0.0

    Env_X = fft(inst_amp_norm - np.mean(inst_amp_norm))
    env_mag_norm = np.abs(Env_X[:len(Env_X)//2])
    env_spec_kurt = kurtosis(env_mag_norm) if len(env_mag_norm) > 3 else 0.0
    env_spec_peak_ratio = np.max(env_mag_norm) / (np.mean(env_mag_norm) + eps)

    modulation_feats = [ia_std, ia_kurt, ia_skew, phase_diff_std, phase_diff_kurt,
                          freq_accel_std, freq_accel_max, env_spec_kurt, env_spec_peak_ratio, 0.0]

    # ================= F. 信息熵类特征 (8) =================
    def permutation_entropy(signal, order=3, delay=1):
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
        return pe / (max_pe + eps)

    def sample_entropy(signal, m=2, r_ratio=0.2):
        signal = signal[::2]
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
                count += np.sum(dist <= r) - 1
                total += len(templates) - 1
            return count / (total + eps)

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        return -np.log(phi_m1 / phi_m + eps)

    perm_entropy = permutation_entropy(x, order=3, delay=1)
    samp_entropy = sample_entropy(x, m=2, r_ratio=0.2)
    perm_entropy_spec = permutation_entropy(mag, order=3, delay=1)

    hist, _ = np.histogram(x, bins=32, density=True)
    hist_norm = hist / (np.sum(hist) + eps)
    shannon_entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + eps))

    approx_entropy_proxy = sample_entropy(x, m=2, r_ratio=0.15)

    entropy_feats = [perm_entropy, samp_entropy, perm_entropy_spec, shannon_entropy,
                      approx_entropy_proxy, 0.0, 0.0, 0.0]

    all_64_feats = (cumulant_feats + wavelet_feats + band_feats +
                     autocorr_feats + modulation_feats + entropy_feats)

    all_64_feats = (all_64_feats + [0.0] * 64)[:64]
    return all_64_feats


FEATURE_NAMES_64_HIGHVALUE = (
    ["|C40|", "|C60|", "C40_norm", "C60_norm", "C63_norm", "|M42|"] +
    [f"Wavelet_EnergyRatio_{i}" for i in range(6)] +
    [f"Wavelet_Std_{i}" for i in range(6)] +
    ["Wavelet_Entropy"] +
    [f"Wavelet_AdjRatio_{i}" for i in range(3)] +
    [f"Band_EnergyRatio_{i}" for i in range(8)] +
    [f"Band_EnergyStd_{i}" for i in range(8)] +
    [f"Autocorr_Lag_{lag}" for lag in [1, 2, 4, 8]] +
    ["Autocorr_FirstZeroLag", "Autocorr_Mean", "Autocorr_Std", "Autocorr_MaxSidelobe"] +
    ["InstAmp_Std", "InstAmp_Kurt", "InstAmp_Skew", "PhaseDiff_Std", "PhaseDiff_Kurt",
     "FreqAccel_Std", "FreqAccel_Max", "EnvSpec_Kurt", "EnvSpec_PeakRatio", "Modulation_Reserved"] +
    ["PermEntropy", "SampleEntropy", "PermEntropy_Spectrum", "ShannonEntropy",
     "ApproxEntropy_Proxy", "Entropy_Reserved1", "Entropy_Reserved2", "Entropy_Reserved3"]
)
assert len(FEATURE_NAMES_64_HIGHVALUE) == 64


# =========================================================
# 4. 统一特征提取接口 (本版本固定输出纯64维高价值特征)
# =========================================================
def extract_features_single(x, fs=1.0):
    """
    纯64维高价值特征，不包含任何原始32维基础统计特征
    """
    cached = compute_cached_intermediates(x, fs=fs)
    highvalue_feats = extract_64_highvalue_features(x, fs, cached)

    all_feats = np.array(highvalue_feats, dtype=np.float64)
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(all_feats) == 64, f"特征维度不匹配: 期望64, 实际{len(all_feats)}"
    return all_feats


def extract_features_batch(X_raw, fs=1.0, tag=""):
    feats = np.zeros((X_raw.shape[0], 64), dtype=np.float64)
    for i in range(X_raw.shape[0]):
        feats[i] = extract_features_single(X_raw[i], fs=fs)
        if (i + 1) % 5000 == 0:
            print(f"  [{tag}] 特征提取进度: {i+1}/{X_raw.shape[0]}")
    return feats


# =========================================================
# 5. LOF 检测器
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
# 6. 评估
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
# 7. 主流程
# =========================================================
def main():
    cfg = Config()
    set_seed(cfg.RANDOM_STATE)

    print(f"当前特征方案: 纯64维高价值特征 (不含原始32维基础特征)")

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
    print(f"\n提取训练集特征 (纯64维高价值特征) ...")
    feat_norm_train_raw = extract_features_batch(X_norm_train, fs=cfg.FS, tag="训练集")

    print("提取正常测试集特征 ...")
    feat_norm_test_raw = extract_features_batch(X_norm_test, fs=cfg.FS, tag="正常测试集")

    print("提取无人机测试集特征 ...")
    feat_drone_test_raw = extract_features_batch(X_drone_test, fs=cfg.FS, tag="无人机测试集")

    # ---------------- 标准化 ----------------
    scaler = StandardScaler()
    feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
    feat_norm_test = scaler.transform(feat_norm_test_raw)
    feat_drone_test = scaler.transform(feat_drone_test_raw)

    # ---------------- 训练LOF ----------------
    print(f"\n训练 LOF 检测器 (纯64维高价值特征) ...")
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
                        title="最终测试结果 - LOF (纯64维高价值特征，不含32维基础特征)")

    analyze_per_drone_model(detector, feat_drone_test, y_drone_test)

    # ---------------- 保存模型 ----------------
    joblib.dump({
        "detector": detector,
        "scaler": scaler,
        "feature_dim": 64,
        "feature_names": FEATURE_NAMES_64_HIGHVALUE
    }, "lof_detector_highvalue64_only.pkl")
    print(f"\n模型已保存至: lof_detector_highvalue64_only.pkl")

    return detector, scaler, metrics


if __name__ == "__main__":
    main()