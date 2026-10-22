"""
Feature-IForest (Isolation Forest) 无人机射频信号异常检测
========================================================
第一级：32维手工特征 + Isolation Forest（性能强于LOF）做异常检测
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
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
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
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # ---- 异常检测器选择 ----
    # "isolation_forest" : 单独使用 Isolation Forest（推荐，速度快，性能强）
    # "ocsvm"            : 单独使用 One-Class SVM
    # "ensemble"         : Isolation Forest + One-Class SVM 分数融合（更稳健）
    DETECTOR_TYPE = "ensemble"

    # Isolation Forest 超参数
    IF_N_ESTIMATORS = 200
    IF_MAX_SAMPLES = "auto"
    IF_CONTAMINATION = "auto"

    # One-Class SVM 超参数（仅在 ocsvm / ensemble 模式下使用）
    OCSVM_KERNEL = "rbf"
    OCSVM_NU = 0.05
    OCSVM_GAMMA = "scale"

    WIFI_DIRS = ["TPLINK_TL_WR940N", "CISCO_LINKSYS_E3200"]
    BT_DIRS = ["APPLE_IPAD3", "APPLE_IPHONE6S", "APPLE_IPHONE7",
               "FITBIT_CHARGE3", "MOTOROLA"]
    DRONE_DIRS = ["DJI_M600", "BEEBEERUN", "DJI_INSPIRE",
                  "DJI_MAVICPRO", "DJI_PHANTOM"]

    FEATURE_NAMES = [
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


def set_seed(seed):
    np.random.seed(seed)


# =========================================================
# 1. 数据加载 (丢弃第1025列，文件夹名为标签)
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
# 2. 32维特征提取（与之前版本一致）
# =========================================================
def extract_32_features(x, fs=1.0):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    eps = 1e-12

    # ---------- 1~8 时域特征 ----------
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

    # ---------- FFT 频谱 ----------
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

    # ---------- STFT 特征 ----------
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

    # ---------- 高阶累积量 ----------
    analytic = sp_signal.hilbert(x)
    z = analytic
    M20 = np.mean(z ** 2)
    M21 = np.mean(np.abs(z) ** 2)
    M40 = np.mean(z ** 4)
    M42 = np.mean(np.abs(z) ** 4)

    C40 = np.abs(M40 - 3 * (M20 ** 2))
    C42 = np.abs(M42 - np.abs(M20) ** 2 - 2 * (M21 ** 2))

    # ---------- 包络 & 瞬时频率 ----------
    envelope = np.abs(analytic)
    envelope_mean = np.mean(envelope)
    envelope_std = np.std(envelope)
    envelope_kurt = kurtosis(envelope)

    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
    inst_freq_std = np.std(inst_freq)

    features = np.array([
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
    ], dtype=np.float64)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_features_batch(X_raw, fs=1.0):
    feats = np.zeros((X_raw.shape[0], 32), dtype=np.float64)
    for i in range(X_raw.shape[0]):
        feats[i] = extract_32_features(X_raw[i], fs=fs)
        if (i + 1) % 2000 == 0:
            print(f"  特征提取进度: {i+1}/{X_raw.shape[0]}")
    return feats


# =========================================================
# 3. 异常检测器（替换 LOF）
# =========================================================
class AnomalyDetector:
    """
    统一封装 Isolation Forest / One-Class SVM / Ensemble
    对外接口:
        fit(X_train)
        predict(X)   -> 1(正常)/ -1(异常)
        score(X)     -> 分数越大越异常
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector_type = cfg.DETECTOR_TYPE
        self.iforest = None
        self.ocsvm = None
        # ---- 新增：固定训练集的归一化统计量 ----
        self.s1_mean_ = None
        self.s1_std_ = None
        self.s2_mean_ = None
        self.s2_std_ = None
        self.ensemble_threshold_ = None

    def fit(self, X_train):
        if self.detector_type in ("isolation_forest", "ensemble"):
            self.iforest = IsolationForest(
                n_estimators=self.cfg.IF_N_ESTIMATORS,
                max_samples=self.cfg.IF_MAX_SAMPLES,
                contamination=self.cfg.IF_CONTAMINATION,
                random_state=self.cfg.RANDOM_STATE,
                n_jobs=-1,
            )
            self.iforest.fit(X_train)

        if self.detector_type in ("ocsvm", "ensemble"):
            self.ocsvm = OneClassSVM(
                kernel=self.cfg.OCSVM_KERNEL,
                nu=self.cfg.OCSVM_NU,
                gamma=self.cfg.OCSVM_GAMMA,
            )
            self.ocsvm.fit(X_train)

        # ---- 关键修复：在训练阶段就固定归一化统计量 ----
        if self.detector_type == "ensemble":
            s1_train = self._iforest_score_raw(X_train)
            s2_train = self._ocsvm_score_raw(X_train)
            self.s1_mean_, self.s1_std_ = s1_train.mean(), s1_train.std() + 1e-12
            self.s2_mean_, self.s2_std_ = s2_train.mean(), s2_train.std() + 1e-12

            # 用训练集自身归一化后的分数确定阈值
            s1_norm = (s1_train - self.s1_mean_) / self.s1_std_
            s2_norm = (s2_train - self.s2_mean_) / self.s2_std_
            train_ensemble_scores = (s1_norm + s2_norm) / 2.0
            self.ensemble_threshold_ = np.percentile(train_ensemble_scores, 99)

        return self

    def _iforest_score_raw(self, X):
        return -self.iforest.decision_function(X)

    def _ocsvm_score_raw(self, X):
        return -self.ocsvm.decision_function(X)

    def score(self, X):
        if self.detector_type == "isolation_forest":
            return self._iforest_score_raw(X)
        elif self.detector_type == "ocsvm":
            return self._ocsvm_score_raw(X)
        elif self.detector_type == "ensemble":
            s1 = self._iforest_score_raw(X)
            s2 = self._ocsvm_score_raw(X)
            # ---- 关键修复：用训练集固定的 mean/std 做归一化，而非当前批次自身的统计量 ----
            s1_norm = (s1 - self.s1_mean_) / self.s1_std_
            s2_norm = (s2 - self.s2_mean_) / self.s2_std_
            return (s1_norm + s2_norm) / 2.0
        else:
            raise ValueError(f"未知的检测器类型: {self.detector_type}")

    def predict(self, X):
        if self.detector_type == "isolation_forest":
            return self.iforest.predict(X)
        elif self.detector_type == "ocsvm":
            return self.ocsvm.predict(X)
        elif self.detector_type == "ensemble":
            scores = self.score(X)
            pred = np.where(scores > self.ensemble_threshold_, -1, 1)
            return pred
        else:
            raise ValueError(f"未知的检测器类型: {self.detector_type}")


def train_detector(feat_train, cfg):
    detector = AnomalyDetector(cfg)
    detector.fit(feat_train)     # 阈值计算已内置在 fit() 中，不再需要额外调用
    return detector


def detect_anomaly(detector, feat):
    pred = detector.predict(feat)     # 1=正常(inlier)，-1=异常(outlier)
    score = detector.score(feat)      # 值越大越可能异常
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

    # ---------- 1. 特征提取 ----------
    print("提取训练集特征 ...")
    feat_norm_train_raw = extract_features_batch(X_norm_train, fs=cfg.FS)
    print("提取正常测试集特征 ...")
    feat_norm_test_raw = extract_features_batch(X_norm_test, fs=cfg.FS)
    print("提取无人机测试集特征 ...")
    feat_drone_raw = extract_features_batch(X_drone, fs=cfg.FS)

    # ---------- 2. 标准化 ----------
    scaler = StandardScaler()
    feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
    feat_norm_test = scaler.transform(feat_norm_test_raw)
    feat_drone = scaler.transform(feat_drone_raw)

    # ---------- 3. 训练异常检测器（Isolation Forest / OCSVM / Ensemble） ----------
    print(f"训练异常检测器: {cfg.DETECTOR_TYPE}")
    detector = train_detector(feat_norm_train, cfg)

    # ---------- 4. 测试推理 ----------
    feat_test_all = np.concatenate([feat_norm_test, feat_drone], axis=0)
    y_test_binary = np.concatenate([
        np.ones(feat_norm_test.shape[0]),
        -np.ones(feat_drone.shape[0]),
    ])

    pred, scores = detect_anomaly(detector, feat_test_all)
    metrics = evaluate(y_test_binary, pred, scores)

    # ---------- 5. (可选) 第二级型号分类 ----------
    if feat_drone.shape[0] > 0:
        drone_clf = train_drone_classifier(feat_drone_raw, y_drone, cfg)
        joblib.dump(drone_clf, "drone_classifier.pkl")

    # ---------- 6. 保存模型 ----------
    joblib.dump({"detector": detector, "scaler": scaler,
                 "feature_names": cfg.FEATURE_NAMES}, "detector_scaler.pkl")

    return detector, scaler, metrics




if __name__ == "__main__":
    main()