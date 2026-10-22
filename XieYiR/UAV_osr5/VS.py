"""
多种主流异常检测算法完整对比实验
========================================================
数据流程: 
  原始CSV -> 加载(丢弃第1025列, 文件夹名为标签) -> 32维手工特征提取 -> 标准化
  -> 训练集(仅WiFi+蓝牙正常信号) 训练多种异常检测器
  -> 测试集(正常测试样本 + 全部无人机样本) 评估

对比模型(13种):
  1.  LOF              (局部离群因子, baseline)
  2.  Isolation Forest (孤立森林)
  3.  One-Class SVM
  4.  ECOD             (经验累积分布函数, 无参数)
  5.  COPOD            (Copula-based, 无参数)
  6.  HBOS             (基于直方图)
  7.  PCA-based        (基于主成分重构误差)
  8.  GMM              (高斯混合模型)
  9.  KDE              (核密度估计)
  10. KNN              (基于K近邻距离)
  11. AutoEncoder      (深度自编码器重构误差)
  12. Deep SVDD        (深度单类支持向量数据描述)
  13. VAE              (变分自编码器)

依赖:
  pip install pyod scikit-learn scipy pandas numpy matplotlib torch
"""

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve)

warnings.filterwarnings("ignore")

# pyod 模型
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA as PyOD_PCA
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.vae import VAE


# =========================================================
# 0. 全局配置
# =========================================================
class Config:
    DATA_ROOT = "./data_root"
    FS = 1.0
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CONTAMINATION = 0.1     # 预估异常比例，用于需要该参数的模型内部阈值计算

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
        data = df.iloc[:, :-1].values.astype(np.float64)
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
# 3. 多模型对比实验核心函数
# =========================================================
def get_all_detectors(cfg, n_features):
    """
    返回所有待对比的检测器字典
    pyod统一约定: predict() -> 0=正常, 1=异常;  decision_function() -> 越大越异常
    """
    detectors = {
        "LOF": LOF(
            n_neighbors=20, contamination=cfg.CONTAMINATION
        ),
        "IForest": IForest(
            n_estimators=200, contamination=cfg.CONTAMINATION,
            random_state=cfg.RANDOM_STATE
        ),
        "OCSVM": OCSVM(
            kernel="rbf", contamination=cfg.CONTAMINATION
        ),
        "ECOD": ECOD(
            contamination=cfg.CONTAMINATION
        ),
        "COPOD": COPOD(
            contamination=cfg.CONTAMINATION
        ),
        "HBOS": HBOS(
            contamination=cfg.CONTAMINATION
        ),
        "PCA": PyOD_PCA(
            contamination=cfg.CONTAMINATION, random_state=cfg.RANDOM_STATE
        ),
        "GMM": GMM(
            n_components=3, contamination=cfg.CONTAMINATION,
            random_state=cfg.RANDOM_STATE
        ),
        "KDE": KDE(
            contamination=cfg.CONTAMINATION
        ),
        "KNN": KNN(
            n_neighbors=20, contamination=cfg.CONTAMINATION
        ),
        "AutoEncoder": AutoEncoder(
            contamination=cfg.CONTAMINATION,
            hidden_neuron_list=[64, 32, 16, 32, 64],
            epoch_num=100, batch_size=256, verbose=0,
            random_state=cfg.RANDOM_STATE
        ),
        "DeepSVDD": DeepSVDD(
            n_features=n_features,
            contamination=cfg.CONTAMINATION,
            hidden_neurons=[64, 32],
            epochs=100, batch_size=256,
            random_state=cfg.RANDOM_STATE
        ),
        "VAE": VAE(
            contamination=cfg.CONTAMINATION,
            encoder_neuron_list=[64, 32, 16],
            decoder_neuron_list=[16, 32, 64],
            latent_dim=8,
            epoch_num=100, batch_size=256, verbose=0,
            random_state=cfg.RANDOM_STATE
        ),
    }
    return detectors


def run_all_detectors(feat_train, feat_test, y_test_binary, cfg):
    """
    feat_train: 训练特征(仅正常样本), 已标准化
    feat_test:  测试特征(正常+无人机混合), 已标准化
    y_test_binary: 1=正常, -1=异常 (我们自定义的约定)

    返回: DataFrame汇总结果, dict{模型名: (pred01, scores)}
    """
    y_test01 = (y_test_binary == -1).astype(int)   # 转换为 pyod 约定: 1=异常, 0=正常

    detectors = get_all_detectors(cfg, n_features=feat_train.shape[1])
    results = []
    raw_outputs = {}   # 保存每个模型的预测结果，供后续绘图使用

    for name, clf in detectors.items():
        print(f"\n{'='*25} 训练模型: {name} {'='*25}")
        t0 = time.time()
        try:
            clf.fit(feat_train)
            train_time = time.time() - t0

            t1 = time.time()
            scores = clf.decision_function(feat_test)   # 越大越异常
            pred01 = clf.predict(feat_test)               # 1=异常, 0=正常
            infer_time = time.time() - t1

            acc = accuracy_score(y_test01, pred01)
            prec = precision_score(y_test01, pred01, zero_division=0)
            rec = recall_score(y_test01, pred01, zero_division=0)
            f1 = f1_score(y_test01, pred01, zero_division=0)
            try:
                auc = roc_auc_score(y_test01, scores)
            except ValueError:
                auc = float("nan")
            cm = confusion_matrix(y_test01, pred01)

            results.append({
                "Model": name, "Accuracy": acc, "Precision": prec,
                "Recall": rec, "F1": f1, "AUC": auc,
                "TrainTime(s)": train_time, "InferTime(s)": infer_time
            })
            raw_outputs[name] = {"pred": pred01, "scores": scores, "cm": cm}

            print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
                  f"F1={f1:.4f}  AUC={auc:.4f}  "
                  f"训练耗时={train_time:.2f}s  推理耗时={infer_time:.2f}s")
            print(f"  混淆矩阵 [0=正常, 1=无人机异常]:\n{cm}")

        except Exception as e:
            print(f"  [失败] {name}: {repr(e)}")
            results.append({
                "Model": name, "Accuracy": np.nan, "Precision": np.nan,
                "Recall": np.nan, "F1": np.nan, "AUC": np.nan,
                "TrainTime(s)": np.nan, "InferTime(s)": np.nan
            })

    df_results = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    return df_results, raw_outputs


# =========================================================
# 4. 结果可视化
# =========================================================
def plot_comparison_bar(df_results, save_path="model_comparison.png"):
    """绘制各模型 Precision/Recall/F1/AUC 对比柱状图"""
    metrics = ["Precision", "Recall", "F1", "AUC"]
    df_plot = df_results.set_index("Model")[metrics]

    fig, ax = plt.subplots(figsize=(14, 6))
    df_plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("异常检测模型性能对比")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"对比柱状图已保存至: {save_path}")
    plt.close()


def plot_roc_curves(raw_outputs, y_test_binary, save_path="roc_curves.png"):
    """绘制所有模型的ROC曲线"""
    y_test01 = (y_test_binary == -1).astype(int)

    plt.figure(figsize=(9, 8))
    for name, output in raw_outputs.items():
        scores = output["scores"]
        try:
            fpr, tpr, _ = roc_curve(y_test01, scores)
            auc_val = roc_auc_score(y_test01, scores)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC曲线对比")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"ROC曲线图已保存至: {save_path}")
    plt.close()


# =========================================================
# 5. 主流程
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
    print(f"\n正常训练样本: {X_norm_train.shape[0]}, 正常测试样本: {X_norm_test.shape[0]}")
    print(f"无人机(全部作为测试期异常样本): {X_drone.shape[0]}")

    # ---------- 特征提取 ----------
    print("\n提取训练集特征 ...")
    feat_norm_train_raw = extract_features_batch(X_norm_train, fs=cfg.FS, tag="训练集")
    print("提取正常测试集特征 ...")
    feat_norm_test_raw = extract_features_batch(X_norm_test, fs=cfg.FS, tag="正常测试集")
    print("提取无人机测试集特征 ...")
    feat_drone_raw = extract_features_batch(X_drone, fs=cfg.FS, tag="无人机测试集")

    # ---------- 标准化 ----------
    scaler = StandardScaler()
    feat_norm_train = scaler.fit_transform(feat_norm_train_raw)
    feat_norm_test = scaler.transform(feat_norm_test_raw)
    feat_drone = scaler.transform(feat_drone_raw)

    feat_test_all = np.concatenate([feat_norm_test, feat_drone], axis=0)
    y_test_binary = np.concatenate([
        np.ones(feat_norm_test.shape[0]),      # 正常 -> 1
        -np.ones(feat_drone.shape[0]),         # 无人机 -> -1
    ])

    print(f"\n测试集总样本数: {feat_test_all.shape[0]} "
          f"(正常: {feat_norm_test.shape[0]}, 无人机: {feat_drone.shape[0]})")

    # ---------- 多模型对比 ----------
    print("\n" + "#" * 70)
    print("开始多模型横向对比实验")
    print("#" * 70)

    df_results, raw_outputs = run_all_detectors(
        feat_norm_train, feat_test_all, y_test_binary, cfg
    )

    print("\n" + "=" * 70)
    print("模型对比汇总 (按F1降序排列):")
    print("=" * 70)
    print(df_results.to_string(index=False))

    df_results.to_csv("detector_comparison_results.csv", index=False, encoding="utf-8-sig")
    print("\n结果已保存至: detector_comparison_results.csv")

    # ---------- 可视化 ----------
    plot_comparison_bar(df_results)
    plot_roc_curves(raw_outputs, y_test_binary)

    # ---------- 输出最优模型详情 ----------
    best_model_name = df_results.iloc[0]["Model"]
    print(f"\n最优模型 (按F1): {best_model_name}")
    print(f"混淆矩阵:\n{raw_outputs[best_model_name]['cm']}")

    return df_results, raw_outputs, scaler


if __name__ == "__main__":
    main()