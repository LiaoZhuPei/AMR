"""
SDAE-LOF 项目 —— 独立化异常检测模块 (LOF / LOF+KNN融合 可切换版)
========================================================================
数据流程 (与多模型对比实验完全一致，不引入无人机校准集):
  原始CSV -> 加载(丢弃第1025列, 文件夹名为标签) -> 32维手工特征提取 -> 标准化
  -> 训练集(仅WiFi+蓝牙正常信号的一部分)  训练检测器 + 基于训练集分布确定阈值
  -> 测试集(正常剩余部分 + 全部无人机)    最终评估，无人机数据全程不参与训练/调参

检测模式可切换:
  DETECTOR_MODE = "lof"              : 单独使用 LOF (pyod实现)
  DETECTOR_MODE = "lof_knn_fusion"   : LOF + KNN 排名融合

阈值确定方式: 纯无监督，基于训练集(正常数据)分数分布的 contamination 分位数
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

from pyod.models.lof import LOF
from pyod.models.knn import KNN

import joblib


# =========================================================
# 0. 全局配置
# =========================================================
class Config:
    DATA_ROOT = "./data_root"
    FS = 1.0                        # 采样率，若已知真实值请修改
    RANDOM_STATE = 42

    # ---- 数据划分 (与多模型对比实验完全一致) ----
    TEST_SIZE = 0.2                 # 正常样本中划出多少做测试，其余全部用于训练
    # 无人机样本不参与划分，全部作为测试期异常样本

    # ---- 检测模式切换 ----
    # "lof"             : 单独使用 LOF
    # "lof_knn_fusion"  : LOF + KNN 排名融合
    DETECTOR_MODE = "lof"

    # ---- 异常比例先验 (仅基于训练集自身分布确定阈值，不使用任何无人机数据) ----
    CONTAMINATION = 0.1             # 与多模型对比实验保持一致的设置

    # ---- LOF 超参数 ----
    LOF_N_NEIGHBORS = 20
    LOF_METRIC = "minkowski"

    # ---- KNN 超参数 (仅fusion模式使用) ----
    KNN_N_NEIGHBORS = 20
    KNN_METRIC = "minkowski"

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
# 2. 32维特征提取
# =========================================================
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


def extract_features_batch(X_raw, fs=1.0, tag=""):
    feats = np.zeros((X_raw.shape[0], 32), dtype=np.float64)
    for i in range(X_raw.shape[0]):
        feats[i] = extract_32_features(X_raw[i], fs=fs)
        if (i + 1) % 5000 == 0:
            print(f"  [{tag}] 特征提取进度: {i+1}/{X_raw.shape[0]}")
    return feats


# =========================================================
# 3. 核心检测器: LOF / LOF+KNN 融合 (统一接口)
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
        """返回异常分数，越大越异常"""
        return self.clf.decision_function(X)

    def predict(self, X):
        """pyod内部阈值(基于训练集分数分布的contamination分位数)判定: 1=异常, 0=正常"""
        return self.clf.predict(X)


class LOFKNNFusionDetector:
    """
    LOF + KNN 排名融合检测器
    核心思路: 分别用LOF和KNN计算异常分数，将各自分数映射为
             "相对于训练集(纯正常数据)分数分布的百分位排名"，
             再取平均作为融合分数。
    阈值确定: 完全基于训练集融合分数分布的 contamination 分位数，
             不使用任何无人机数据（与LOF单模型的阈值确定逻辑保持一致）
    统一接口: fit(X_train) / score(X) [越大越异常] / predict(X) [1=异常, 0=正常]
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.lof_clf = LOF(
            n_neighbors=cfg.LOF_N_NEIGHBORS,
            metric=cfg.LOF_METRIC,
            contamination=cfg.CONTAMINATION,
        )
        self.knn_clf = KNN(
            n_neighbors=cfg.KNN_N_NEIGHBORS,
            metric=cfg.KNN_METRIC,
            contamination=cfg.CONTAMINATION,
        )
        self.lof_train_scores_sorted_ = None
        self.knn_train_scores_sorted_ = None
        self.threshold_ = None   # 基于训练集融合分数分布确定，无监督

    def fit(self, X_train):
        self.lof_clf.fit(X_train)
        self.knn_clf.fit(X_train)

        # 固定训练集分数分布，用于后续百分位排名映射
        lof_train_scores = self.lof_clf.decision_function(X_train)
        knn_train_scores = self.knn_clf.decision_function(X_train)
        self.lof_train_scores_sorted_ = np.sort(lof_train_scores)
        self.knn_train_scores_sorted_ = np.sort(knn_train_scores)

        # ---- 关键: 阈值完全基于训练集(纯正常数据)融合分数分布确定 ----
        # 计算训练集自身的融合分数(百分位排名)
        lof_train_pct = self._to_percentile(self.lof_train_scores_sorted_, lof_train_scores)
        knn_train_pct = self._to_percentile(self.knn_train_scores_sorted_, knn_train_scores)
        train_fused_scores = (lof_train_pct + knn_train_pct) / 2.0

        # 用 contamination 分位数作为阈值 (与pyod内部逻辑保持一致的无监督方式)
        self.threshold_ = np.percentile(train_fused_scores, 100 * (1 - self.cfg.CONTAMINATION))

        return self

    def _to_percentile(self, sorted_train_scores, scores):
        """将任意输入的分数，映射为相对于(固定的)训练集分数分布的百分位排名(0~1)"""
        percentiles = np.searchsorted(sorted_train_scores, scores) / len(sorted_train_scores)
        return percentiles

    def score(self, X):
        """融合分数 = LOF百分位排名 与 KNN百分位排名 的平均值，越大越异常"""
        lof_scores = self.lof_clf.decision_function(X)
        knn_scores = self.knn_clf.decision_function(X)

        lof_pct = self._to_percentile(self.lof_train_scores_sorted_, lof_scores)
        knn_pct = self._to_percentile(self.knn_train_scores_sorted_, knn_scores)

        fused = (lof_pct + knn_pct) / 2.0
        return fused

    def predict(self, X):
        """使用fit阶段基于训练集分布确定的阈值判定，不涉及任何无人机数据"""
        scores = self.score(X)
        return (scores > self.threshold_).astype(int)


def build_detector(cfg):
    """根据配置构建对应的检测器"""
    if cfg.DETECTOR_MODE == "lof":
        return LOFDetector(cfg)
    elif cfg.DETECTOR_MODE == "lof_knn_fusion":
        return LOFKNNFusionDetector(cfg)
    else:
        raise ValueError(f"未知的 DETECTOR_MODE: {cfg.DETECTOR_MODE}")


# =========================================================
# 4. 评估
# =========================================================
def evaluate(y_true_binary, pred01, scores, title=""):
    """
    y_true_binary: 1=正常, -1=异常 (自定义约定)
    pred01: 1=异常, 0=正常 (与pyod一致)
    scores: 异常分数，越大越异常
    """
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


# =========================================================
# 5. 按无人机型号细分析 (辅助诊断，仅用于分析，不参与训练/阈值确定)
# =========================================================
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
    print("各无人机型号检测召回率明细 (仅用于分析诊断，按recall升序):")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)

    return summary


# =========================================================
# 6. 主流程
# =========================================================
def main():
    cfg = Config()
    set_seed(cfg.RANDOM_STATE)

    print(f"当前检测模式: {cfg.DETECTOR_MODE}")

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

    # ---------------- 数据划分 (与多模型对比实验完全一致) ----------------
    # 正常样本: 训练 / 测试 两段划分
    X_norm_train, X_norm_test = train_test_split(
        X_normal, test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE, shuffle=True
    )
    # 无人机样本: 全部作为测试集异常样本，不参与训练，不用于任何阈值确定
    X_drone_test, y_drone_test = X_drone, y_drone_model

    print(f"\n数据划分结果 (与多模型对比实验一致):")
    print(f"  正常-训练集: {X_norm_train.shape[0]}")
    print(f"  正常-测试集: {X_norm_test.shape[0]}")
    print(f"  无人机-测试集(全部,不参与训练/调参): {X_drone_test.shape[0]}")

    # ---------------- 特征提取 ----------------
    print("\n提取训练集特征 ...")
    feat_norm_train_raw = extract_features_batch(X_norm_train, fs=cfg.FS, tag="训练集")

    print("提取正常测试集特征 ...")
    feat_norm_test_raw = extract_features_batch(X_norm_test, fs=cfg.FS, tag="正常测试集")

    print("提取无人机测试集特征 ...")
    feat_drone_test_raw = extract_features_batch(X_drone_test, fs=cfg.FS, tag="无人机测试集")

    # ---------------- 标准化 (仅用训练集拟合scaler) ----------------
    scaler = StandardScaler()
    feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
    feat_norm_test = scaler.transform(feat_norm_test_raw)
    feat_drone_test = scaler.transform(feat_drone_test_raw)

    # ---------------- 训练检测器 (阈值在fit内部基于训练集自身分布确定) ----------------
    print(f"\n训练检测器: {cfg.DETECTOR_MODE} ...")
    detector = build_detector(cfg)
    detector.fit(feat_norm_train)

    # ---------------- 最终测试评估 ----------------
    feat_test_all = np.concatenate([feat_norm_test, feat_drone_test], axis=0)
    y_test_binary = np.concatenate([
        np.ones(feat_norm_test.shape[0]),      # 正常 -> 1
        -np.ones(feat_drone_test.shape[0]),    # 无人机 -> -1
    ])

    scores_test = detector.score(feat_test_all)
    pred_test = detector.predict(feat_test_all)

    metrics = evaluate(y_test_binary, pred_test, scores_test,
                        title=f"最终测试结果 - {cfg.DETECTOR_MODE}")

    # ---------------- 按无人机型号细分析 (仅诊断用途) ----------------
    analyze_per_drone_model(detector, feat_drone_test, y_drone_test)

    # ---------------- 保存模型 ----------------
    joblib.dump({
        "detector": detector,
        "scaler": scaler,
        "detector_mode": cfg.DETECTOR_MODE,
        "feature_names": cfg.FEATURE_NAMES
    }, f"detector_{cfg.DETECTOR_MODE}.pkl")
    print(f"\n模型已保存至: detector_{cfg.DETECTOR_MODE}.pkl")

    return detector, scaler, metrics


# =========================================================
# 7. 两种模式对比运行 (可选：一次性跑完两种模式做对比)
# =========================================================
def run_both_modes_and_compare():
    """依次运行 lof 和 lof_knn_fusion 两种模式，输出对比表格"""
    cfg_base = Config()
    set_seed(cfg_base.RANDOM_STATE)

    # ---------------- 数据加载与划分 (两种模式共用同一份划分，保证公平对比) ----------------
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

    print("提取特征 ...")
    feat_norm_train_raw = extract_features_batch(X_norm_train, fs=cfg_base.FS, tag="训练集")
    feat_norm_test_raw = extract_features_batch(X_norm_test, fs=cfg_base.FS, tag="正常测试集")
    feat_drone_test_raw = extract_features_batch(X_drone_test, fs=cfg_base.FS, tag="无人机测试集")

    scaler = StandardScaler()
    feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
    feat_norm_test = scaler.transform(feat_norm_test_raw)
    feat_drone_test = scaler.transform(feat_drone_test_raw)

    feat_test_all = np.concatenate([feat_norm_test, feat_drone_test], axis=0)
    y_test_binary = np.concatenate([
        np.ones(feat_norm_test.shape[0]), -np.ones(feat_drone_test.shape[0])
    ])

    all_metrics = {}
    for mode in ["lof", "lof_knn_fusion"]:
        print("\n" + "#" * 70)
        print(f"运行模式: {mode}")
        print("#" * 70)

        cfg = Config()
        cfg.DETECTOR_MODE = mode

        detector = build_detector(cfg)
        detector.fit(feat_norm_train)

        scores_test = detector.score(feat_test_all)
        pred_test = detector.predict(feat_test_all)
        metrics = evaluate(y_test_binary, pred_test, scores_test, title=mode)

        analyze_per_drone_model(detector, feat_drone_test, y_drone_test)

        all_metrics[mode] = metrics

    print("\n" + "=" * 70)
    print("两种检测模式对比汇总:")
    print("=" * 70)
    df_compare = pd.DataFrame(all_metrics).T[["accuracy", "precision", "recall", "f1", "auc"]]
    print(df_compare.to_string())
    df_compare.to_csv("lof_vs_fusion_comparison.csv")

    return df_compare


if __name__ == "__main__":
    # 方式1: 只运行 Config 中当前配置的单一模式
    main()

    # 方式2: 若想一次性对比两种模式，注释掉上面 main()，改用下面这行
    # run_both_modes_and_compare()