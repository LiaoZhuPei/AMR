# pip install numpy scipy scikit-learn pandas matplotlib umap-learn tqdm h5py
import os, re, glob, time, traceback
import numpy as np
import h5py
import scipy.io as sio
from scipy.signal import stft
from scipy.ndimage import label as cc_label
from scipy.stats import ks_2samp, kurtosis, entropy, skew
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, roc_auc_score,
                              average_precision_score, roc_curve,
                              accuracy_score, confusion_matrix, classification_report)
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

FS = 80e6
FC = 2.44e9
N_SAMPLES = int(FS * 50e-3)  # 4,000,000


# ============================================================
# 1. 文件解析与IO (兼容 v7.3/HDF5)
# ============================================================

def parse_filename(fname):
    base = os.path.basename(fname)
    name_no_ext = os.path.splitext(base)[0]
    if len(name_no_ext) == 0:
        raise ValueError(f"无法解析文件名: {base}")

    mtype = name_no_ext[0].upper()

    indiv_match = re.match(r'^[A-Za-z](\d+)', name_no_ext)
    indiv = indiv_match.group(1) if indiv_match else '0'

    cond_match = re.search(r'(IN|OUT)', name_no_ext, re.IGNORECASE)
    cond = cond_match.group(1).upper() if cond_match else 'UNK'

    seq_match = re.search(r'S(\d+)', name_no_ext, re.IGNORECASE)
    seq = seq_match.group(1) if seq_match else '0'

    slice_match = re.search(r'slice[_-]?(\d+)', name_no_ext, re.IGNORECASE)
    slice_ = slice_match.group(1) if slice_match else '0'

    return {'mtype': mtype, 'indiv': indiv, 'cond': cond, 'seq': seq, 'slice': slice_}


def load_iq(fname):
    if h5py.is_hdf5(fname):
        with h5py.File(fname, 'r') as f:
            I = np.asarray(f['I'], dtype=np.float32).squeeze()
            Q = np.asarray(f['Q'], dtype=np.float32).squeeze()
    else:
        mat = sio.loadmat(fname)
        I = np.asarray(mat['I']).astype(np.float32).squeeze()
        Q = np.asarray(mat['Q']).astype(np.float32).squeeze()
    return I + 1j * Q


def scan_dataset(root_dir):
    files = glob.glob(os.path.join(root_dir, '*.mat'))
    records = []
    for f in files:
        try:
            meta = parse_filename(f)
        except ValueError:
            continue
        meta['file'] = f
        records.append(meta)
    return pd.DataFrame(records)


# ============================================================
# 2. 时频谱预处理
# ============================================================

STFT_NPERSEG = 2048
STFT_NOVERLAP = int(STFT_NPERSEG * 0.75)


def compute_spectrogram(x, fs=FS, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP):
    f, t, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                      return_onesided=False, boundary=None)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    Pxx = np.abs(Zxx) ** 2
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    return f, t, Pxx_db


def noise_floor_normalize(Sxx_db, pct=10):
    noise_floor = np.percentile(Sxx_db, pct, axis=1, keepdims=True)
    return Sxx_db - noise_floor, noise_floor.squeeze()


def preprocess(x):
    f, t, Sxx_db = compute_spectrogram(x)
    Sxx_norm, noise_floor = noise_floor_normalize(Sxx_db)
    return f, t, Sxx_norm, noise_floor


def _safe_skew(arr):
    return skew(arr) if len(arr) > 2 else 0.0


# ============================================================
# 3. 协议层统计特征
# ============================================================

def extract_videoband_features(f, t, Sxx_norm, thresh_db=6.0, min_band_width_hz=1e5):
    n_freq, n_time = Sxx_norm.shape
    active = Sxx_norm > thresh_db
    occ = active.mean(axis=1)

    occ_active = occ > 0.15
    labeled, n_bands = cc_label(occ_active)

    freq_res = f[1] - f[0]
    band_widths, band_centers, band_energy_ratio = [], [], []
    total_energy = np.sum(10 ** (Sxx_norm / 10))

    for b in range(1, n_bands + 1):
        idx = np.where(labeled == b)[0]
        width = len(idx) * freq_res
        if width < min_band_width_hz:
            continue
        center = f[idx].mean()
        energy = np.sum(10 ** (Sxx_norm[idx, :] / 10))
        band_widths.append(width)
        band_centers.append(center)
        band_energy_ratio.append(energy / (total_energy + 1e-12))

    band_widths = np.array(band_widths)
    band_centers = np.sort(np.array(band_centers))
    gaps = np.diff(band_centers) if len(band_centers) > 1 else np.array([0.0])

    def stat_moments(arr, name):
        if len(arr) == 0:
            arr = np.array([0.0])
        return {
            f'{name}_mean': arr.mean(), f'{name}_std': arr.std(),
            f'{name}_min': arr.min(), f'{name}_max': arr.max(),
            f'{name}_skew': _safe_skew(arr), f'{name}_kurt': kurtosis(arr) if len(arr) > 1 else 0.0
        }

    feats = {'n_video_bands': len(band_widths)}
    feats.update(stat_moments(band_widths, 'band_width'))
    feats.update(stat_moments(gaps, 'band_gap'))
    feats.update(stat_moments(np.array(band_energy_ratio), 'band_energy'))
    feats['video_energy_ratio'] = float(np.sum(band_energy_ratio))
    feats['occ_profile_entropy'] = entropy(occ / (occ.sum() + 1e-12) + 1e-12)
    return feats


def extract_hopping_burst_features(f, t, Sxx_norm, video_band_mask=None,
                                    power_thresh_db=8.0, max_burst_dur=2e-3):
    n_freq, n_time = Sxx_norm.shape
    if video_band_mask is None:
        video_band_mask = np.ones(n_freq, dtype=bool)

    lin_power = 10 ** (Sxx_norm / 10)
    wideband_power = lin_power[video_band_mask, :].sum(axis=0)
    wb_db = 10 * np.log10(wideband_power + 1e-12)
    thresh = np.median(wb_db) + power_thresh_db
    active = wb_db > thresh

    dt = t[1] - t[0] if len(t) > 1 else 1.0
    labeled, n_seg = cc_label(active)

    burst_durations, burst_centroids = [], []
    for s in range(1, n_seg + 1):
        idx = np.where(labeled == s)[0]
        dur = len(idx) * dt
        if dur > max_burst_dur:
            continue
        seg_spec = lin_power[:, idx].sum(axis=1)
        centroid = np.sum(f * seg_spec) / (np.sum(seg_spec) + 1e-12)
        burst_durations.append(dur)
        burst_centroids.append(centroid)

    burst_durations = np.array(burst_durations)
    burst_centroids = np.array(burst_centroids)
    freq_jumps = np.abs(np.diff(burst_centroids)) if len(burst_centroids) > 1 else np.array([0.0])

    total_time = t[-1] - t[0] if len(t) > 1 else 1.0
    feats = {
        'burst_rate': len(burst_durations) / total_time,
        'burst_dur_mean': burst_durations.mean() if len(burst_durations) else 0.0,
        'burst_dur_std': burst_durations.std() if len(burst_durations) else 0.0,
        'freq_jump_mean': freq_jumps.mean(),
        'freq_jump_std': freq_jumps.std(),
        'freq_jump_max': freq_jumps.max(),
        'hopping_bw_range': (burst_centroids.max() - burst_centroids.min()) if len(burst_centroids) > 1 else 0.0
    }
    return feats


def extract_interference_features(f, t, Sxx_norm, video_freq_range):
    f_low, f_high = video_freq_range
    below_mask = f < f_low

    lin_power = 10 ** (Sxx_norm / 10)
    total_energy = lin_power.sum()
    below_energy = lin_power[below_mask, :].sum()

    active_below = (Sxx_norm[below_mask, :] > 6.0)
    active_ratio = active_below.mean() if active_below.size > 0 else 0.0

    avg_psd = lin_power.mean(axis=1)
    p = avg_psd / (avg_psd.sum() + 1e-12)
    spec_entropy = entropy(p + 1e-12)

    return {
        'below_band_energy_ratio': below_energy / (total_energy + 1e-12),
        'below_band_active_ratio': active_ratio,
        'global_spectral_entropy': spec_entropy
    }


def extract_statistical_features(x, f, avg_psd=None, Sxx_norm=None):
    if avg_psd is None:
        lin_power = 10 ** (Sxx_norm / 10)
        avg_psd = lin_power.mean(axis=1)

    p = avg_psd / (avg_psd.sum() + 1e-12)
    centroid = np.sum(f * p)
    bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * p))
    flatness = np.exp(np.mean(np.log(avg_psd + 1e-12))) / (np.mean(avg_psd) + 1e-12)
    spec_kurt = kurtosis(avg_psd)

    xn = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    m20 = np.mean(xn ** 2)
    m21 = np.mean(xn * np.conj(xn))
    m40 = np.mean(xn ** 4)
    m41 = np.mean((xn ** 3) * np.conj(xn))
    m42 = np.mean((xn ** 2) * (np.conj(xn) ** 2))
    C40 = m40 - 3 * m20 ** 2
    C41 = m41 - 3 * m20 * m21
    C42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2

    return {
        'spec_centroid': centroid, 'spec_bandwidth': bandwidth,
        'spec_flatness': flatness, 'spec_kurtosis': spec_kurt,
        'C20_abs': np.abs(m20), 'C40_abs': np.abs(C40),
        'C41_abs': np.abs(C41), 'C42_abs': np.abs(C42)
    }


# ============================================================
# 4. 硬件级射频指纹统计特征
# ============================================================

def extract_hardware_fingerprint_features(x, fs=FS):
    feats = {}
    I = x.real.astype(np.float64)
    Q = x.imag.astype(np.float64)

    std_I = np.std(I) + 1e-12
    std_Q = np.std(Q) + 1e-12
    feats['iq_amp_imbalance'] = std_I / std_Q

    cross = np.mean(I * Q)
    denom = np.mean(I ** 2) + np.mean(Q ** 2) + 1e-12
    feats['iq_phase_imbalance'] = np.arcsin(np.clip(2 * cross / denom, -1, 1))

    dc_I, dc_Q = np.mean(I), np.mean(Q)
    feats['dc_offset_I'] = dc_I
    feats['dc_offset_Q'] = dc_Q
    feats['dc_offset_mag'] = np.sqrt(dc_I ** 2 + dc_Q ** 2)

    phase = np.unwrap(np.angle(x))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    lo, hi = np.percentile(inst_freq, [1, 99])
    inst_freq_c = np.clip(inst_freq, lo, hi)
    sub = inst_freq_c[::100]
    feats['inst_freq_mean'] = inst_freq_c.mean()
    feats['inst_freq_std'] = inst_freq_c.std()
    feats['inst_freq_skew'] = _safe_skew(sub)
    feats['inst_freq_kurt'] = kurtosis(sub) if len(sub) > 1 else 0.0

    env = np.abs(x)
    env_power = env ** 2
    mean_p = env_power.mean() + 1e-12
    peak_p = env_power.max()
    feats['papr_db'] = 10 * np.log10(peak_p / mean_p + 1e-12)
    feats['env_kurtosis'] = kurtosis(env)
    feats['env_skew'] = _safe_skew(env[::100])
    feats['env_cv'] = env.std() / (env.mean() + 1e-12)

    dec = 8
    env_detrend = (env_power - env_power.mean())[::dec]
    fs_ds = fs / dec
    n = len(env_detrend)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(env_detrend * win))
    freqs = np.fft.rfftfreq(n, d=1 / fs_ds)
    mask = freqs > 1e3
    spec_m, freqs_m = spec[mask], freqs[mask]

    if len(spec_m) > 0:
        top_idx = np.argsort(spec_m)[-3:][::-1]
        top_freqs = freqs_m[top_idx]
        top_ratio = spec_m[top_idx] / (spec_m.sum() + 1e-12)
    else:
        top_freqs, top_ratio = np.zeros(3), np.zeros(3)

    for i in range(3):
        feats[f'env_spec_peak{i+1}_freq'] = float(top_freqs[i]) if i < len(top_freqs) else 0.0
        feats[f'env_spec_peak{i+1}_ratio'] = float(top_ratio[i]) if i < len(top_ratio) else 0.0

    return feats


def extract_all_features(fname, video_freq_range_global=(-2e7, 2e7)):
    x = load_iq(fname)
    f, t, Sxx_norm, noise_floor = preprocess(x)

    feats = {}
    feats.update(extract_videoband_features(f, t, Sxx_norm))
    feats.update(extract_hopping_burst_features(f, t, Sxx_norm))
    feats.update(extract_interference_features(f, t, Sxx_norm, video_freq_range_global))
    feats.update(extract_statistical_features(x, f, Sxx_norm=Sxx_norm))
    feats.update(extract_hardware_fingerprint_features(x))
    return feats


def build_feature_table(df_meta, save_path='feat_partial.csv', save_every=100):
    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(tqdm(df_meta.iterrows(), total=len(df_meta))):
        t1 = time.time()
        try:
            feats = extract_all_features(r['file'])
        except Exception as e:
            print(f"跳过 {r['file']}: {e}")
            traceback.print_exc()
            continue
        dt = time.time() - t1
        if dt > 5:
            print(f"[慢样本] {r['file']} 用时 {dt:.2f}s")
        feats.update(r.to_dict())
        rows.append(feats)

        if (i + 1) % save_every == 0:
            pd.DataFrame(rows).to_csv(save_path, index=False)
            print(f"已处理 {i+1}/{len(df_meta)}, 累计耗时 {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"全部完成, 总耗时 {time.time()-t0:.1f}s")
    return df


# ============================================================
# 5. Known/Unknown 划分
# ============================================================

KNOWN_CLASSES = ['A', 'D', 'E', 'G']
UNKNOWN_CLASSES = ['B', 'C', 'F']


def prepare_known_unknown(feat_df, feature_cols):
    known_df = feat_df[feat_df['mtype'].isin(KNOWN_CLASSES)].copy()
    unknown_df = feat_df[feat_df['mtype'].isin(UNKNOWN_CLASSES)].copy()

    scaler = StandardScaler()
    scaler.fit(known_df[feature_cols].values)

    known_df[feature_cols] = scaler.transform(known_df[feature_cols].values)
    unknown_df[feature_cols] = scaler.transform(unknown_df[feature_cols].values)
    return known_df, unknown_df, scaler


# ============================================================
# 6. 类内/类间分析
# ============================================================

def intra_class_compactness(known_df, feature_cols):
    results = {}
    S_W_pooled = 0.0
    for c in KNOWN_CLASSES:
        Xc = known_df.loc[known_df['mtype'] == c, feature_cols].values
        mu = Xc.mean(axis=0)
        d2 = np.sum((Xc - mu) ** 2, axis=1)
        results[c] = {'mean_dist2': d2.mean(), 'std_dist2': d2.std(), 'n_samples': len(Xc)}
        S_W_pooled += d2.sum()
    return results, S_W_pooled


def inter_class_separation(known_df, feature_cols):
    X = known_df[feature_cols].values
    y = known_df['mtype'].values
    k = len(KNOWN_CLASSES)
    n = len(X)

    mu_global = X.mean(axis=0)
    S_B = 0.0
    for c in KNOWN_CLASSES:
        Xc = known_df.loc[known_df['mtype'] == c, feature_cols].values
        mu_c = Xc.mean(axis=0)
        S_B += len(Xc) * np.sum((mu_c - mu_global) ** 2)

    _, S_W = intra_class_compactness(known_df, feature_cols)
    fisher_J = (S_B / (k - 1)) / (S_W / (n - k) + 1e-12)

    sil = silhouette_score(X, y)
    ch = calinski_harabasz_score(X, y)
    db = davies_bouldin_score(X, y)

    centers = {c: known_df.loc[known_df['mtype'] == c, feature_cols].values.mean(axis=0)
               for c in KNOWN_CLASSES}
    center_dist = pd.DataFrame(
        [[np.linalg.norm(centers[a] - centers[b]) for b in KNOWN_CLASSES] for a in KNOWN_CLASSES],
        index=KNOWN_CLASSES, columns=KNOWN_CLASSES)

    return {'fisher_J(=CH校验)': fisher_J, 'silhouette': sil,
            'calinski_harabasz': ch, 'davies_bouldin': db,
            'center_dist_matrix': center_dist}


# ============================================================
# 7. 稳定性分析
# ============================================================

def rbf_mmd2(X, Y, gamma=None):
    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        Z = np.vstack([X, Y])
        from scipy.spatial.distance import pdist
        gamma = 1.0 / (2 * np.median(pdist(Z)) ** 2 + 1e-12)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def stability_individual(known_df, feature_cols):
    results = {}
    for c in KNOWN_CLASSES:
        sub = known_df[known_df['mtype'] == c]
        mu_c = sub[feature_cols].values.mean(axis=0)
        S_W_indiv, S_B_indiv = 0.0, 0.0
        for indiv in sub['indiv'].unique():
            Xi = sub.loc[sub['indiv'] == indiv, feature_cols].values
            mu_i = Xi.mean(axis=0)
            S_W_indiv += np.sum((Xi - mu_i) ** 2)
            S_B_indiv += len(Xi) * np.sum((mu_i - mu_c) ** 2)
        results[c] = S_B_indiv / (S_W_indiv + 1e-12)
    return results


def stability_condition(known_df, feature_cols):
    results = {}
    for c in KNOWN_CLASSES:
        sub = known_df[known_df['mtype'] == c]
        X_in = sub.loc[sub['cond'] == 'IN', feature_cols].values
        X_out = sub.loc[sub['cond'] == 'OUT', feature_cols].values
        results[c] = rbf_mmd2(X_in, X_out) if len(X_in) > 1 and len(X_out) > 1 else np.nan
    return results


def stability_time(known_df, feature_cols):
    results = {}
    for c in KNOWN_CLASSES:
        sub = known_df[known_df['mtype'] == c]
        mu_c = sub[feature_cols].values.mean(axis=0)
        S_W, S_B = 0.0, 0.0
        for seq in sub['seq'].unique():
            Xs = sub.loc[sub['seq'] == seq, feature_cols].values
            mu_s = Xs.mean(axis=0)
            S_W += np.sum((Xs - mu_s) ** 2)
            S_B += len(Xs) * np.sum((mu_s - mu_c) ** 2)
        results[c] = S_B / (S_W + 1e-12)
    return results


def overall_ICC(known_df, feature_cols):
    iccs = []
    groups = known_df.groupby('mtype')
    grand_mean = known_df[feature_cols].values.mean(axis=0)
    k = len(KNOWN_CLASSES)
    for j, col in enumerate(feature_cols):
        ss_between, ss_within, n_total = 0.0, 0.0, 0
        for c, g in groups:
            vals = g[col].values
            n_total += len(vals)
            ss_between += len(vals) * (vals.mean() - grand_mean[j]) ** 2
            ss_within += np.sum((vals - vals.mean()) ** 2)
        ms_between = ss_between / (k - 1 + 1e-12)
        ms_within = ss_within / (n_total - k + 1e-12)
        n_bar = n_total / k
        icc = (ms_between - ms_within) / (ms_between + (n_bar - 1) * ms_within + 1e-12)
        iccs.append(icc)
    return np.mean(iccs), dict(zip(feature_cols, iccs))


def select_features_by_icc(known_df, feature_cols, icc_threshold=0.05, min_keep=10):
    _, icc_detail = overall_ICC(known_df, feature_cols)
    icc_series = pd.Series(icc_detail).sort_values(ascending=False)
    selected = icc_series[icc_series > icc_threshold].index.tolist()

    print(f"\n[特征选择] 原始特征数={len(feature_cols)}, ICC>{icc_threshold}保留={len(selected)}")
    print("ICC Top15:\n", icc_series.head(15))

    if len(selected) < min_keep:
        print(f"警告: 满足阈值的特征不足{min_keep}个, 回退为ICC最高的Top{min_keep}")
        selected = icc_series.index.tolist()[:min_keep]
    return selected, icc_series


# ============================================================
# 8. 【核心新增】按录音分组的严格交叉验证 (防数据泄露)
# ============================================================

def make_group_id(df):
    """
    以 机型+个体+采集条件+时间序号 作为一次"原始录音"的唯一标识
    同一次录音切出的所有slice必须分到同一折, 不能一部分训练一部分验证
    """
    return (df['mtype'].astype(str) + '_' +
            df['indiv'].astype(str) + '_' +
            df['cond'].astype(str) + '_' +
            df['seq'].astype(str)).values


def supervised_classification_baseline(known_df, feature_cols, n_splits=5,
                                        use_group_cv=True, verbose_name=""):
    """
    use_group_cv=True  -> 按录音分组的GroupKFold(防泄露, 更可信)
    use_group_cv=False -> 普通StratifiedKFold(可能存在同录音跨折泄露, 仅作对比参照)
    """
    X = known_df[feature_cols].values
    y = known_df['mtype'].values

    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=0)
    }

    results = {}
    cv_tag = "GroupKFold(按录音分组, 防泄露)" if use_group_cv else "StratifiedKFold(普通, 可能泄露)"
    print(f"\n{'='*20} {verbose_name} | CV方式: {cv_tag} {'='*20}")

    if use_group_cv:
        groups = make_group_id(known_df)
        n_groups = len(np.unique(groups))
        actual_splits = min(n_splits, n_groups)
        if actual_splits < n_splits:
            print(f"警告: 分组数({n_groups})少于设定折数({n_splits}), 自动降为{actual_splits}折")
        cv = GroupKFold(n_splits=actual_splits)
        split_iter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        split_iter = cv.split(X, y)

    split_iter = list(split_iter)  # 复用两次(两个分类器)

    for name, clf in classifiers.items():
        y_pred = np.empty_like(y)
        for train_idx, val_idx in split_iter:
            clf_fold = clf.__class__(**clf.get_params())
            clf_fold.fit(X[train_idx], y[train_idx])
            y_pred[val_idx] = clf_fold.predict(X[val_idx])

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=KNOWN_CLASSES)
        print(f"\n--- {name} ---")
        print(f"整体准确率: {acc:.4f}")
        print("混淆矩阵(行=真实,列=预测):")
        print(pd.DataFrame(cm, index=KNOWN_CLASSES, columns=KNOWN_CLASSES))
        print(classification_report(y, y_pred, labels=KNOWN_CLASSES, zero_division=0))
        results[name] = {'accuracy': acc, 'confusion_matrix': cm}

    rf_full = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf_full.fit(X, y)
    importances = pd.Series(rf_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n[{verbose_name} | RandomForest 特征重要性 Top15]\n", importances.head(15))

    results['rf_full_model'] = rf_full
    results['feature_importances'] = importances
    return results


def compare_cv_methods(known_df, feature_cols, verbose_name=""):
    """
    同时跑 StratifiedKFold(可能泄露) 和 GroupKFold(防泄露), 对比准确率差异
    这是判断"之前结果是否虚高"的核心诊断步骤
    """
    res_strat = supervised_classification_baseline(
        known_df, feature_cols, use_group_cv=False, verbose_name=verbose_name)
    res_group = supervised_classification_baseline(
        known_df, feature_cols, use_group_cv=True, verbose_name=verbose_name)

    print(f"\n{'#'*60}")
    print(f"[{verbose_name}] StratifiedKFold vs GroupKFold 对比")
    print(f"{'#'*60}")
    diffs = {}
    for clf_name in ['RandomForest', 'SVM_RBF']:
        acc_strat = res_strat[clf_name]['accuracy']
        acc_group = res_group[clf_name]['accuracy']
        gap = acc_strat - acc_group
        diffs[clf_name] = gap
        flag = "⚠️ 差距较大,存在泄露风险" if gap > 0.10 else ("轻微差距,基本可信" if gap > 0.03 else "几乎一致,无明显泄露")
        print(f"{clf_name}: StratifiedKFold={acc_strat:.4f}  GroupKFold={acc_group:.4f}  "
              f"差距={gap:.4f}  [{flag}]")

    return res_strat, res_group, diffs


# ============================================================
# 9. 开放集检测
# ============================================================

def open_set_with_classifier(clf, known_df, unknown_df, feature_cols, fpr_target=0.05):
    X_known = known_df[feature_cols].values
    X_unknown_all = unknown_df[feature_cols].values

    score_known = 1 - clf.predict_proba(X_known).max(axis=1)
    score_unknown_all = 1 - clf.predict_proba(X_unknown_all).max(axis=1)
    y_known = np.zeros(len(score_known))

    results = {}
    for uc in UNKNOWN_CLASSES:
        mask = unknown_df['mtype'].values == uc
        score_u = score_unknown_all[mask]
        y_all = np.concatenate([y_known, np.ones(len(score_u))])
        score_all = np.concatenate([score_known, score_u])

        auroc = roc_auc_score(y_all, score_all)
        aupr = average_precision_score(y_all, score_all)
        fpr, tpr, thr = roc_curve(y_all, score_all)
        idx = np.argmin(np.abs(fpr - fpr_target))
        results[uc] = {'AUROC': auroc, 'AUPR': aupr, f'TPR@FPR={fpr_target}': tpr[idx]}

    y_all = np.concatenate([y_known, np.ones(len(score_unknown_all))])
    score_all = np.concatenate([score_known, score_unknown_all])
    results['overall'] = {'AUROC': roc_auc_score(y_all, score_all),
                           'AUPR': average_precision_score(y_all, score_all)}
    return results


def open_set_with_classifier_groupcv(known_df, unknown_df, feature_cols, n_splits=5, fpr_target=0.05):
    """
    用GroupKFold方式获得known样本的'诚实'置信度(out-of-fold预测),
    而不是用同一份数据训练+预测(那样known的置信度会被高估, 让AUROC失真)
    unknown类完全没在训练里出现过, 不受影响, 但known这边也应公平对待
    """
    X = known_df[feature_cols].values
    y = known_df['mtype'].values
    groups = make_group_id(known_df)
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=actual_splits)

    oof_score_known = np.zeros(len(y))
    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        oof_score_known[val_idx] = 1 - clf.predict_proba(X[val_idx]).max(axis=1)

    # unknown检测用全量known训练的模型(unknown类本来就没参与训练, 无泄露问题)
    clf_full = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf_full.fit(X, y)
    X_unknown_all = unknown_df[feature_cols].values
    score_unknown_all = 1 - clf_full.predict_proba(X_unknown_all).max(axis=1)

    y_known = np.zeros(len(oof_score_known))
    results = {}
    for uc in UNKNOWN_CLASSES:
        mask = unknown_df['mtype'].values == uc
        score_u = score_unknown_all[mask]
        y_all = np.concatenate([y_known, np.ones(len(score_u))])
        score_all = np.concatenate([oof_score_known, score_u])

        auroc = roc_auc_score(y_all, score_all)
        aupr = average_precision_score(y_all, score_all)
        fpr, tpr, thr = roc_curve(y_all, score_all)
        idx = np.argmin(np.abs(fpr - fpr_target))
        results[uc] = {'AUROC': auroc, 'AUPR': aupr, f'TPR@FPR={fpr_target}': tpr[idx]}

    y_all = np.concatenate([y_known, np.ones(len(score_unknown_all))])
    score_all = np.concatenate([oof_score_known, score_unknown_all])
    results['overall'] = {'AUROC': roc_auc_score(y_all, score_all),
                           'AUPR': average_precision_score(y_all, score_all)}
    results['clf_full_model'] = clf_full
    return results


def unknown_distance_analysis(known_df, unknown_df, feature_cols):
    centers = {c: known_df.loc[known_df['mtype'] == c, feature_cols].values.mean(axis=0)
               for c in KNOWN_CLASSES}
    cov_pooled = LedoitWolf().fit(known_df[feature_cols].values).covariance_
    cov_inv = np.linalg.pinv(cov_pooled)

    def maha_min(x):
        return min(np.sqrt((x - mu) @ cov_inv @ (x - mu)) for mu in centers.values())

    known_intra_dist = []
    for c in KNOWN_CLASSES:
        Xc = known_df.loc[known_df['mtype'] == c, feature_cols].values
        for x in Xc:
            known_intra_dist.append(np.sqrt((x - centers[c]) @ cov_inv @ (x - centers[c])))
    known_intra_dist = np.array(known_intra_dist)

    result = {}
    for uc in UNKNOWN_CLASSES:
        Xu = unknown_df.loc[unknown_df['mtype'] == uc, feature_cols].values
        dists = np.array([maha_min(x) for x in Xu])
        ks_stat, ks_p = ks_2samp(known_intra_dist, dists)
        result[uc] = {'mean_dist': dists.mean(), 'ks_stat_vs_known_intra': ks_stat, 'ks_pvalue': ks_p}
    result['known_intra_dist'] = known_intra_dist
    return result


def known_unknown_separability(known_df, unknown_df, feature_cols, fpr_target=0.05):
    centers = {c: known_df.loc[known_df['mtype'] == c, feature_cols].values.mean(axis=0)
               for c in KNOWN_CLASSES}
    cov_pooled = LedoitWolf().fit(known_df[feature_cols].values).covariance_
    cov_inv = np.linalg.pinv(cov_pooled)

    def maha_min(x):
        return min(np.sqrt((x - mu) @ cov_inv @ (x - mu)) for mu in centers.values())

    X_known = known_df[feature_cols].values
    score_known = np.array([maha_min(x) for x in X_known])
    y_known = np.zeros(len(score_known))

    results = {}
    for uc in UNKNOWN_CLASSES:
        Xu = unknown_df.loc[unknown_df['mtype'] == uc, feature_cols].values
        score_u = np.array([maha_min(x) for x in Xu])
        y_all = np.concatenate([y_known, np.ones(len(score_u))])
        score_all = np.concatenate([score_known, score_u])
        results[uc] = {'AUROC': roc_auc_score(y_all, score_all),
                        'AUPR': average_precision_score(y_all, score_all)}

    Xu_all = unknown_df[feature_cols].values
    score_u_all = np.array([maha_min(x) for x in Xu_all])
    y_all = np.concatenate([y_known, np.ones(len(score_u_all))])
    score_all = np.concatenate([score_known, score_u_all])
    results['overall'] = {'AUROC': roc_auc_score(y_all, score_all),
                           'AUPR': average_precision_score(y_all, score_all)}
    return results


# ============================================================
# 10. 最终判决 (以GroupKFold结果为准)
# ============================================================

def final_decision_report(acc_group, auroc_group_openset, auroc_maha,
                           acc_threshold=0.70, auroc_threshold=0.80):
    print("\n" + "=" * 70)
    print("最终评估 (以GroupKFold按录音分组的结果为准, 排除数据泄露干扰)")
    print("=" * 70)

    print(f"已知类分类准确率(GroupKFold): {acc_group:.4f}  (参考阈值: {acc_threshold})")
    print(f"未知类检测AUROC(GroupKFold-OOF分类器置信度法): {auroc_group_openset:.4f}  (参考阈值: {auroc_threshold})")
    print(f"未知类检测AUROC(马氏距离法, 仅供参照): {auroc_maha:.4f}")

    if acc_group >= acc_threshold and auroc_group_openset >= auroc_threshold:
        verdict = ("统计特征基本可用(经防泄露验证): 已知类可分性和开放集检测能力均达标, "
                   "建议继续在统计特征基础上调优, 深度学习非必需。")
    elif acc_group >= acc_threshold and auroc_group_openset < auroc_threshold:
        verdict = ("已知机型分类没问题, 但开放集检测能力经严格验证后不足, "
                   "建议优先尝试专门的OOD方法(能量得分/OpenMax/单类SVM), "
                   "若仍不达标再考虑深度学习+专用OOD头部。")
    else:
        verdict = ("经过防泄露的严格验证后, 统计特征对已知机型的判别力不足, "
                   "说明之前的高准确率主要来自数据泄露(记住了同一段录音的特性), "
                   "真实泛化能力有限, 建议转向深度学习(时频谱图CNN / 端到端IQ序列模型)。")

    print("\n>>> 结论:", verdict)
    return verdict


# ============================================================
# 11. 可视化
# ============================================================

def visualize_feature_space(known_df, unknown_df, feature_cols, method='tsne'):
    from sklearn.manifold import TSNE
    X_known = known_df[feature_cols].values
    X_unknown = unknown_df[feature_cols].values
    X_all = np.vstack([X_known, X_unknown])
    labels = list(known_df['mtype']) + list(unknown_df['mtype'])

    n_samples = X_all.shape[0]
    if n_samples < 4:
        print(f"样本数太少({n_samples}), 跳过可视化")
        return

    if method == 'tsne':
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        perplexity = min(perplexity, n_samples - 1)
        emb = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(X_all)
    else:
        import umap
        emb = umap.UMAP(n_components=2, random_state=0).fit_transform(X_all)

    plt.figure(figsize=(8, 6))
    colors = {'A': 'C0', 'D': 'C1', 'E': 'C2', 'G': 'C3',
              'B': 'gray', 'C': 'black', 'F': 'brown'}
    for cls in KNOWN_CLASSES + UNKNOWN_CLASSES:
        idx = [i for i, l in enumerate(labels) if l == cls]
        marker = 'o' if cls in KNOWN_CLASSES else 'x'
        plt.scatter(emb[idx, 0], emb[idx, 1], label=cls,
                    c=colors.get(cls, 'k'), marker=marker, alpha=0.7)
    plt.legend()
    plt.title(f'Known(圆点) vs Unknown(叉) 特征空间分布 ({method})')
    plt.show()


# ============================================================
# 12. 主流程
# ============================================================

if __name__ == '__main__':
    ROOT = 'E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train'
    DEBUG = False
    FORCE_RECOMPUTE = False

    FEAT_CSV = 'feat_debug_v2.csv' if DEBUG else 'feat_all_v2.csv'

    df_meta = scan_dataset(ROOT)
    print(f"共扫描到 {len(df_meta)} 个样本, 类别分布:\n{df_meta['mtype'].value_counts()}")

    if DEBUG:
        df_meta = df_meta.groupby('mtype').head(2).reset_index(drop=True)
        print(f"[DEBUG模式] 仅使用 {len(df_meta)} 个样本")

    if os.path.exists(FEAT_CSV) and not FORCE_RECOMPUTE:
        print(f"检测到已有特征文件 {FEAT_CSV}, 直接加载")
        feat_df = pd.read_csv(FEAT_CSV)
    else:
        print(f"开始提取特征(样本数={len(df_meta)})...")
        feat_df = build_feature_table(df_meta, save_path=FEAT_CSV.replace('.csv', '_partial.csv'))
        feat_df.to_csv(FEAT_CSV, index=False)

    feature_cols = [c for c in feat_df.columns
                    if c not in ['mtype', 'indiv', 'cond', 'seq', 'slice', 'file']]
    print(f"\n特征总数: {len(feature_cols)}")

    known_df, unknown_df, scaler = prepare_known_unknown(feat_df, feature_cols)

    # 检查一下分组数量, 让你心里有数(比如每个机型有多少段不同的原始录音)
    known_df['_group'] = make_group_id(known_df)
    print("\n[分组统计] 每个机型的唯一录音段数量(用于GroupKFold):")
    print(known_df.groupby('mtype')['_group'].nunique())

    # ---- 1. 类内/类间 ----
    intra_res, S_W = intra_class_compactness(known_df, feature_cols)
    print("\n[类内紧凑性]", intra_res, "\nS_W(pooled) =", S_W)

    inter_res = inter_class_separation(known_df, feature_cols)
    print("\n[类间分离度]", {k: v for k, v in inter_res.items() if k != 'center_dist_matrix'})
    print(inter_res['center_dist_matrix'])

    # ---- 2. 稳定性 ----
    print("\n[个体稳定性 Fisher比]", stability_individual(known_df, feature_cols))
    print("[IN/OUT MMD^2]", stability_condition(known_df, feature_cols))
    print("[时间序号稳定性]", stability_time(known_df, feature_cols))
    mean_icc, icc_detail = overall_ICC(known_df, feature_cols)
    print("[整体类别ICC]", mean_icc)

    # ---- 3. 【核心】GroupKFold vs StratifiedKFold 对比: 全特征 ----
    res_strat_full, res_group_full, diffs_full = compare_cv_methods(
        known_df, feature_cols, verbose_name="全部特征")

    # ---- 4. 特征选择后再对比一次(可选, 供参考) ----
    selected_cols, icc_series = select_features_by_icc(known_df, feature_cols, icc_threshold=0.05)
    res_strat_sel, res_group_sel, diffs_sel = compare_cv_methods(
        known_df, selected_cols, verbose_name="ICC筛选特征")

    # 以GroupKFold结果为准挑选更优特征集
    if res_group_sel['RandomForest']['accuracy'] >= res_group_full['RandomForest']['accuracy']:
        best_cols = selected_cols
        best_group_res = res_group_sel
        print("\n>>> [以GroupKFold为准] 筛选后特征效果更优(或持平), 后续使用筛选特征集")
    else:
        best_cols = feature_cols
        best_group_res = res_group_full
        print("\n>>> [以GroupKFold为准] 全特征效果更优, 后续使用全部特征")

    best_acc_group = max(best_group_res['RandomForest']['accuracy'],
                          best_group_res['SVM_RBF']['accuracy'])

    # ---- 5. 开放集检测: 用GroupKFold的out-of-fold置信度(防泄露的诚实评估) ----
    open_set_group = open_set_with_classifier_groupcv(known_df, unknown_df, best_cols)
    print("\n[开放集检测-分类器置信度法(GroupKFold-OOF, 防泄露)]")
    for uc in UNKNOWN_CLASSES + ['overall']:
        print(f"  {uc}: AUROC={open_set_group[uc]['AUROC']:.3f}, AUPR={open_set_group[uc]['AUPR']:.3f}")

    # 对比: 普通(可能泄露)方式的开放集结果, 仅作参照
    rf_full_leaky = res_strat_full['rf_full_model']
    open_set_leaky = open_set_with_classifier(rf_full_leaky, known_df, unknown_df, feature_cols)
    print("\n[开放集检测-分类器置信度法(普通训练集直接预测, 可能偏乐观, 仅参照)]")
    for uc in UNKNOWN_CLASSES + ['overall']:
        print(f"  {uc}: AUROC={open_set_leaky[uc]['AUROC']:.3f}, AUPR={open_set_leaky[uc]['AUPR']:.3f}")

    # 马氏距离法(仅参照)
    open_set_maha = known_unknown_separability(known_df, unknown_df, best_cols)
    print("\n[开放集检测-马氏距离法(仅参照)]")
    for uc in UNKNOWN_CLASSES + ['overall']:
        print(f"  {uc}: AUROC={open_set_maha[uc]['AUROC']:.3f}, AUPR={open_set_maha[uc]['AUPR']:.3f}")

    # ---- 6. 最终判决(以GroupKFold结果为唯一权威依据) ----
    final_decision_report(
        acc_group=best_acc_group,
        auroc_group_openset=open_set_group['overall']['AUROC'],
        auroc_maha=open_set_maha['overall']['AUROC']
    )

    # ---- 7. 可视化 ----
    visualize_feature_space(known_df, unknown_df, best_cols, method='tsne')