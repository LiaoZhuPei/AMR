"""
drone_rf_feature_analysis.py
无人机射频信号特征提取与特征质量分析（Known: A/D/E/G, Unknown: B/C/F）
标签仅为文件名首字母，不做任何个体编号解析。
"""

import os
import glob
import json
import warnings
import argparse

import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

KNOWN_CLASSES = ['A', 'D', 'E', 'G']
UNKNOWN_CLASSES = ['B', 'C', 'F']
FS = 80e6  # 采样率


# =====================================================================
# 1. 数据读取
# =====================================================================

def load_iq_mat_v73(filepath):
    """读取 MATLAB v7.3（HDF5）格式的 .mat 文件，提取 I/Q 分量并组成复数信号"""
    with h5py.File(filepath, 'r') as f:
        if 'I' not in f or 'Q' not in f:
            raise KeyError(f"{filepath} 中未找到 I/Q 字段，实际字段: {list(f.keys())}")
        I = np.array(f['I']).squeeze().astype(np.float32)
        Q = np.array(f['Q']).squeeze().astype(np.float32)
    x = I.astype(np.float64) + 1j * Q.astype(np.float64)
    return x


def parse_label(filename):
    """标签 = 文件名首字母（忽略大小写），仅此而已，不做任何编号解析"""
    base = os.path.basename(filename)
    return base[0].upper()


# =====================================================================
# 2. 特征提取（与之前一致）
# =====================================================================

def spectral_features(x, fs, nperseg=8192):
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg,
                               noverlap=nperseg // 2,
                               return_onesided=False, scaling='density')
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    psd_norm = psd / (np.sum(psd) + 1e-20)

    centroid = np.sum(freqs * psd_norm)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-20)) / np.log2(len(psd_norm))

    gm = np.exp(np.mean(np.log(psd + 1e-20)))
    am = np.mean(psd)
    flatness = gm / (am + 1e-20)

    peaks, _ = signal.find_peaks(psd, prominence=np.max(psd) * 0.05)
    peak_count = len(peaks)
    if peak_count >= 2:
        peak_freqs = np.sort(freqs[peaks])
        spacing = np.diff(peak_freqs)
        spacing_mean, spacing_std = spacing.mean(), spacing.std()
    else:
        spacing_mean, spacing_std = 0.0, 0.0

    cdf = np.cumsum(psd_norm)
    idx_low = np.searchsorted(cdf, 0.005)
    idx_high = np.searchsorted(cdf, 0.995)
    occ_bw = freqs[min(idx_high, len(freqs) - 1)] - freqs[max(idx_low, 0)]

    papr = np.max(psd) / (am + 1e-20)

    return {
        'spec_centroid': centroid,
        'spec_bandwidth': bandwidth,
        'spec_entropy': spec_entropy,
        'spec_flatness': flatness,
        'peak_count': peak_count,
        'peak_spacing_mean': spacing_mean,
        'peak_spacing_std': spacing_std,
        'occupied_bw': occ_bw,
        'spec_papr': papr,
    }


def time_domain_features(x):
    amp = np.abs(x)
    power = amp ** 2
    return {
        'amp_mean': np.mean(amp),
        'amp_std': np.std(amp),
        'amp_skew': stats.skew(amp),
        'amp_kurt': stats.kurtosis(amp),
        'papr_time': np.max(power) / (np.mean(power) + 1e-20),
        'i_kurt': stats.kurtosis(x.real),
        'q_kurt': stats.kurtosis(x.imag),
    }


def instantaneous_freq_features(x, fs):
    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    return {
        'ifreq_std': np.std(inst_freq),
        'ifreq_skew': stats.skew(inst_freq),
        'ifreq_kurt': stats.kurtosis(inst_freq),
        'ifreq_max_jump': np.max(np.abs(np.diff(inst_freq))),
    }


def time_frequency_features(x, fs, nperseg=2048, noverlap=1024, energy_pct=0.7):
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                             return_onesided=False)
    S = np.abs(Zxx) ** 2
    order = np.argsort(f)
    f, S = f[order], S[order, :]

    thresh = np.percentile(S, 100 * energy_pct)
    active_mask = S > thresh
    duty_cycle = np.mean(active_mask)

    active_bands_per_frame = np.sum(active_mask, axis=0)
    active_bands_mean = np.mean(active_bands_per_frame)
    active_bands_std = np.std(active_bands_per_frame)

    dominant_idx = np.argmax(S, axis=0)
    dominant_freq = f[dominant_idx]
    freq_track_std = np.std(dominant_freq)
    freq_track_range = np.max(dominant_freq) - np.min(dominant_freq)

    freq_jumps = np.abs(np.diff(dominant_freq))
    hop_count = np.sum(freq_jumps > 1e6)
    time_span = t[-1] - t[0] if len(t) > 1 else 1.0
    hop_rate = hop_count / time_span

    energy_env = np.sum(S, axis=0)
    env_thresh = np.percentile(energy_env, 100 * energy_pct)
    burst_binary = (energy_env > env_thresh).astype(int)
    diffs = np.diff(np.concatenate(([0], burst_binary, [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    burst_lengths = ends - starts
    dt = (t[1] - t[0]) if len(t) > 1 else 1.0
    if len(burst_lengths) > 0:
        burst_count = len(burst_lengths)
        burst_dur_mean = np.mean(burst_lengths) * dt
        burst_dur_std = np.std(burst_lengths) * dt
    else:
        burst_count, burst_dur_mean, burst_dur_std = 0, 0.0, 0.0

    return {
        'duty_cycle': duty_cycle,
        'active_bands_mean': active_bands_mean,
        'active_bands_std': active_bands_std,
        'freq_track_std': freq_track_std,
        'freq_track_range': freq_track_range,
        'hop_rate': hop_rate,
        'burst_count': burst_count,
        'burst_dur_mean': burst_dur_mean,
        'burst_dur_std': burst_dur_std,
    }


def extract_features(x, fs=FS):
    feats = {}
    feats.update(spectral_features(x, fs))
    feats.update(time_domain_features(x))
    feats.update(instantaneous_freq_features(x, fs))
    feats.update(time_frequency_features(x, fs))
    return feats


# =====================================================================
# 3. 构建数据集（不再解析个体编号，只有 label）
# =====================================================================

def build_dataset(root_dir, fs=FS, verbose=True):
    filepaths = sorted(glob.glob(os.path.join(root_dir, '*.mat')))
    records = []
    for i, fp in enumerate(filepaths):
        label = parse_label(fp)
        try:
            x = load_iq_mat_v73(fp)
            feats = extract_features(x, fs)
        except Exception as e:
            print(f"[跳过] {fp}: {e}")
            continue
        feats['label'] = label
        feats['filepath'] = fp
        records.append(feats)
        if verbose and (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(filepaths)}")
    return pd.DataFrame(records)


# =====================================================================
# 4. 特征质量分析
# =====================================================================

def intra_class_compactness(X, labels):
    """类内紧凑性：样本到类中心距离越小越好"""
    results = {}
    for c in np.unique(labels):
        Xc = X[labels == c]
        centroid = Xc.mean(axis=0)
        dists = np.linalg.norm(Xc - centroid, axis=1)
        results[c] = {
            'mean_dist_to_centroid': float(dists.mean()),
            'std_dist_to_centroid': float(dists.std()),
            'mean_pairwise_dist': float(cdist(Xc, Xc).mean()) if len(Xc) > 1 else 0.0
        }
    return results


def inter_class_separation(X, labels):
    """类间分离度：中心距离矩阵 + Fisher判别比 + Silhouette"""
    classes = np.unique(labels)
    centroids = {c: X[labels == c].mean(axis=0) for c in classes}
    sep_matrix = pd.DataFrame(index=classes, columns=classes, dtype=float)
    for c1 in classes:
        for c2 in classes:
            sep_matrix.loc[c1, c2] = np.linalg.norm(centroids[c1] - centroids[c2])

    overall_mean = X.mean(axis=0)
    Sb, Sw = 0.0, 0.0
    for c in classes:
        Xc = X[labels == c]
        nc = len(Xc)
        mc = Xc.mean(axis=0)
        Sb += nc * np.sum((mc - overall_mean) ** 2)
        Sw += np.sum((Xc - mc) ** 2)
    fisher_ratio = Sb / (Sw + 1e-20)

    sil = silhouette_score(X, labels) if len(classes) > 1 else np.nan
    return sep_matrix, fisher_ratio, sil


def feature_stability_kfold(X, labels, n_splits=5, random_state=42):
    """
    特征稳定性（不依赖任何个体编号）：
    对每个类别的样本做 K 折随机划分，计算每折子集中心 与 全类中心 的偏差，
    stability = 1 - fold间中心方差 / 类内总方差
    越接近1，说明该类特征在不同随机子集下越稳定，不依赖具体是哪些样本。
    """
    report = {}
    for c in np.unique(labels):
        Xc = X[labels == c]
        n = len(Xc)
        if n < n_splits:
            report[c] = np.nan
            continue
        class_centroid = Xc.mean(axis=0)
        total_var = np.mean(np.sum((Xc - class_centroid) ** 2, axis=1))

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_centroids = []
        for _, idx in kf.split(Xc):
            fold_centroids.append(Xc[idx].mean(axis=0))
        fold_centroids = np.array(fold_centroids)

        between_fold_var = np.mean(np.sum((fold_centroids - class_centroid) ** 2, axis=1))
        stability = 1 - between_fold_var / (total_var + 1e-20)
        report[c] = float(stability)
    return report


def kfold_known_unknown_distance(X_known, y_known, X_unknown, n_splits=5, random_state=42):
    """
    K折交叉验证下：
    - 已知类留出折样本 到 其真实类中心 的马氏距离（作为"类内基准距离"）
    - 未知类样本 到 最近已知类中心 的马氏距离（每折取均值）
    完全基于随机K折，不依赖任何个体编号信息。
    """
    classes = np.unique(y_known)
    kf_per_class = {c: list(KFold(n_splits=n_splits, shuffle=True,
                                   random_state=random_state).split(
                                       np.arange(np.sum(y_known == c))))
                     for c in classes}
    idx_per_class = {c: np.where(y_known == c)[0] for c in classes}

    known_self_dists = []
    unknown_dists_folds = []

    for fold in range(n_splits):
        centroids, inv_covs = {}, {}
        test_indices_global = {}

        for c in classes:
            train_local, test_local = kf_per_class[c][fold]
            global_idx = idx_per_class[c]
            train_idx = global_idx[train_local]
            test_idx = global_idx[test_local]
            test_indices_global[c] = test_idx

            Xc_train = X_known[train_idx]
            centroids[c] = Xc_train.mean(axis=0)
            cov = LedoitWolf().fit(Xc_train).covariance_
            inv_covs[c] = np.linalg.pinv(cov)

        for c in classes:
            for s in X_known[test_indices_global[c]]:
                diff = s - centroids[c]
                d = np.sqrt(diff @ inv_covs[c] @ diff.T)
                known_self_dists.append(d)

        fold_unknown = []
        for s in X_unknown:
            ds = [np.sqrt((s - centroids[c]) @ inv_covs[c] @ (s - centroids[c]).T)
                  for c in classes]
            fold_unknown.append(np.min(ds))
        unknown_dists_folds.append(fold_unknown)

    known_self_dists = np.array(known_self_dists)
    unknown_dists = np.mean(np.array(unknown_dists_folds), axis=0)
    return known_self_dists, unknown_dists


def known_unknown_separability(known_dists, unknown_dists):
    """以马氏距离作为OOD分数，计算 Known/Unknown 可分性 AUROC"""
    y_true = np.concatenate([np.zeros_like(known_dists), np.ones_like(unknown_dists)])
    y_score = np.concatenate([known_dists, unknown_dists])
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc, fpr, tpr


# =====================================================================
# 5. 可视化
# =====================================================================

def visualize_features(X_known, y_known, X_unknown, y_unknown, output_dir):
    X_all = np.vstack([X_known, X_unknown])
    y_all = np.concatenate([y_known, y_unknown])
    classes = np.unique(y_all)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    def _plot(X2d, title, fname):
        plt.figure(figsize=(8, 6))
        for c, col in zip(classes, colors):
            mask = y_all == c
            marker = 'x' if c in UNKNOWN_CLASSES else 'o'
            plt.scatter(X2d[mask, 0], X2d[mask, 1], label=c,
                        color=col, marker=marker, alpha=0.7)
        plt.legend()
        plt.title(title)
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

    pca = PCA(n_components=2)
    _plot(pca.fit_transform(X_all), 'PCA 特征分布 (o=已知类, x=未知类)', 'pca_visualization.png')

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    _plot(tsne.fit_transform(X_all), 't-SNE 特征分布 (o=已知类, x=未知类)', 'tsne_visualization.png')


def plot_roc(fpr, tpr, auc, output_dir):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Known/Unknown 可分性 ROC 曲线')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'roc_known_unknown.png'), dpi=150)
    plt.close()


def plot_distance_distribution(known_d, unknown_d, output_dir):
    plt.figure(figsize=(7, 5))
    plt.hist(known_d, bins=30, alpha=0.6, label='已知类(留出测试)距离')
    plt.hist(unknown_d, bins=30, alpha=0.6, label='未知类距离')
    plt.xlabel('马氏距离')
    plt.ylabel('样本数')
    plt.legend()
    plt.title('已知/未知类到最近已知类中心的距离分布')
    plt.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=150)
    plt.close()


# =====================================================================
# 6. 主流程
# =====================================================================

def main(train_dir, output_dir='results', use_cache=True):
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, 'features_raw.csv')

    if use_cache and os.path.exists(cache_path):
        print(f"检测到缓存特征文件 {cache_path}，直接加载...")
        df = pd.read_csv(cache_path)
    else:
        print("Step 1: 加载数据并提取特征 ...")
        df = build_dataset(train_dir)
        df.to_csv(cache_path, index=False)

    feature_cols = [c for c in df.columns if c not in ('label', 'filepath')]
    X_all = df[feature_cols].values.astype(np.float64)
    y_all = df['label'].values.astype(str)

    known_mask = np.isin(y_all, KNOWN_CLASSES)
    unknown_mask = np.isin(y_all, UNKNOWN_CLASSES)

    X_known_raw, y_known = X_all[known_mask], y_all[known_mask]
    X_unknown_raw, y_unknown = X_all[unknown_mask], y_all[unknown_mask]

    # 标准化：只用已知类拟合，防止未知类信息泄露
    scaler = StandardScaler().fit(X_known_raw)
    X_known = scaler.transform(X_known_raw)
    X_unknown = scaler.transform(X_unknown_raw)

    report = {}

    print("\nStep 2: 类内紧凑性分析 ...")
    compactness = intra_class_compactness(X_known, y_known)
    report['intra_class_compactness'] = compactness
    for c, v in compactness.items():
        print(f"  类别 {c}: 到中心均距={v['mean_dist_to_centroid']:.3f}, "
              f"类内样本对均距={v['mean_pairwise_dist']:.3f}")

    print("\nStep 3: 类间分离度分析 ...")
    sep_matrix, fisher_ratio, sil = inter_class_separation(X_known, y_known)
    print(sep_matrix.round(3))
    print(f"  Fisher判别比: {fisher_ratio:.3f}   Silhouette系数: {sil:.3f}")
    report['inter_class_separation'] = {
        'distance_matrix': sep_matrix.to_dict(),
        'fisher_ratio': float(fisher_ratio),
        'silhouette_score': float(sil)
    }

    print("\nStep 4: 特征稳定性分析（K折随机重采样） ...")
    stability = feature_stability_kfold(X_known, y_known)
    report['feature_stability'] = stability
    for c, v in stability.items():
        print(f"  类别 {c}: 稳定性得分={v:.3f}")

    print("\nStep 5: 未知类距离 & Known/Unknown 可分性分析 (K折) ...")
    known_self_d, unknown_d = kfold_known_unknown_distance(X_known, y_known, X_unknown)
    print(f"  已知类留出测试样本到真实类中心马氏距离: mean={known_self_d.mean():.3f}, std={known_self_d.std():.3f}")
    print(f"  未知类样本到最近已知类中心马氏距离: mean={unknown_d.mean():.3f}, std={unknown_d.std():.3f}")

    per_class_unknown = {}
    for c in np.unique(y_unknown):
        d = unknown_d[y_unknown == c]
        per_class_unknown[c] = {'mean': float(d.mean()), 'std': float(d.std())}
        print(f"    未知类 {c}: mean_dist={d.mean():.3f}")

    auc, fpr, tpr = known_unknown_separability(known_self_d, unknown_d)
    print(f"  Known/Unknown 可分性 AUROC = {auc:.4f}")

    report['unknown_distance'] = {
        'known_self_dist_mean': float(known_self_d.mean()),
        'known_self_dist_std': float(known_self_d.std()),
        'unknown_dist_mean': float(unknown_d.mean()),
        'unknown_dist_std': float(unknown_d.std()),
        'per_class_unknown': per_class_unknown
    }
    report['known_unknown_auroc'] = float(auc)

    print("\nStep 6: 可视化 ...")
    visualize_features(X_known, y_known, X_unknown, y_unknown, output_dir)
    plot_roc(fpr, tpr, auc, output_dir)
    plot_distance_distribution(known_self_d, unknown_d, output_dir)

    with open(os.path.join(output_dir, 'quality_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n全部完成，结果已保存至: {output_dir}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train', required=True, help='train文件夹路径')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--no_cache', action='store_true')
    args = parser.parse_args()

    main(args.train_dir, args.output_dir, use_cache=not args.no_cache)