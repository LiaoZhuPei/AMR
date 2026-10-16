# pip install numpy scipy scikit-learn pandas matplotlib umap-learn tqdm h5py
import os, re, glob
import numpy as np
import h5py
import scipy.io as sio
from scipy.signal import stft
from scipy.ndimage import label as cc_label
from scipy.stats import ks_2samp, kurtosis, entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, roc_auc_score,
                              average_precision_score, roc_curve)
from sklearn.covariance import LedoitWolf
import pandas as pd
import matplotlib.pyplot as plt

FS = 80e6
FC = 2.44e9
N_SAMPLES = int(FS * 50e-3)  # 4,000,000

from tqdm import tqdm
import time
import traceback

def build_feature_table(df_meta, save_path='feat_partial.csv', save_every=50):
    """df_meta 来自 scan_dataset(), 逐文件提取特征并拼成大表
       - 加入 tqdm 进度条
       - 每处理 save_every 个样本落盘一次, 防止中途崩溃/中断丢失已算结果
       - 记录每个文件耗时, 方便定位异常慢的样本
    """
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
        if dt > 5:  # 超过5秒的文件打印出来, 方便定位慢在哪
            print(f"[慢样本] {r['file']} 用时 {dt:.2f}s")
        feats.update(r.to_dict())
        rows.append(feats)

        if (i + 1) % save_every == 0:
            pd.DataFrame(rows).to_csv(save_path, index=False)
            print(f"已处理 {i+1}/{len(df_meta)}, 累计耗时 {time.time()-t0:.1f}s, 已保存中间结果")

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"全部完成, 总耗时 {time.time()-t0:.1f}s")
    return df

def parse_filename(fname):
    """
    解析文件名, 返回类别/个体/采集条件/时间序号/片段序号

    简化规则:
      - mtype: 文件名首字母, 不区分大小写, 统一转大写
               (例如 Background_xxx.mat -> 'B', a1_IN_S0_slice_3.mat -> 'A')
      - 其余字段(indiv/cond/seq/slice)采用宽松提取, 提取不到就给默认值,
        不再对文件名整体格式做强匹配校验
    """
    base = os.path.basename(fname)
    name_no_ext = os.path.splitext(base)[0]

    if len(name_no_ext) == 0:
        raise ValueError(f"无法解析文件名: {base}")

    mtype = name_no_ext[0].upper()

    # 个体号: 类别字母后面紧跟的数字(如果有)
    indiv_match = re.match(r'^[A-Za-z](\d+)', name_no_ext)
    indiv = indiv_match.group(1) if indiv_match else '0'

    # 采集条件 IN/OUT, 不区分大小写
    cond_match = re.search(r'(IN|OUT)', name_no_ext, re.IGNORECASE)
    cond = cond_match.group(1).upper() if cond_match else 'UNK'

    # 时间序号 S<digits>
    seq_match = re.search(r'S(\d+)', name_no_ext, re.IGNORECASE)
    seq = seq_match.group(1) if seq_match else '0'

    # 片段序号 slice_<digits>
    slice_match = re.search(r'slice[_-]?(\d+)', name_no_ext, re.IGNORECASE)
    slice_ = slice_match.group(1) if slice_match else '0'

    return {'mtype': mtype, 'indiv': indiv, 'cond': cond,
            'seq': seq, 'slice': slice_}


def load_iq(fname):
    """
    读取 .mat 文件, 返回复数基带信号
    兼容:
      - 普通 MATLAB 格式 (<= v7.2, scipy.io.loadmat 可直接读取)
      - MATLAB v7.3 格式 (本质是 HDF5, 需要用 h5py 读取)
    """
    try:
        mat = sio.loadmat(fname)
        I = np.asarray(mat['I']).astype(np.float32).squeeze()
        Q = np.asarray(mat['Q']).astype(np.float32).squeeze()
    except NotImplementedError:
        # v7.3 (HDF5) 格式, scipy 会抛出 NotImplementedError
        with h5py.File(fname, 'r') as f:
            I = np.asarray(f['I'], dtype=np.float32).squeeze()
            Q = np.asarray(f['Q'], dtype=np.float32).squeeze()
    except OSError:
        # 部分环境下 v7.3 文件用 scipy 读取会直接触发 OSError, 兜底再走 h5py
        with h5py.File(fname, 'r') as f:
            I = np.asarray(f['I'], dtype=np.float32).squeeze()
            Q = np.asarray(f['Q'], dtype=np.float32).squeeze()

    x = I + 1j * Q
    return x


def scan_dataset(root_dir):
    """扫描train目录, 返回文件列表及元信息DataFrame"""
    files = glob.glob(os.path.join(root_dir, '*.mat'))
    records = []
    for f in files:
        try:
            meta = parse_filename(f)
        except ValueError:
            continue
        meta['file'] = f
        records.append(meta)
    df = pd.DataFrame(records)
    return df


STFT_NPERSEG = 2048
STFT_NOVERLAP = int(STFT_NPERSEG * 0.75)


def compute_spectrogram(x, fs=FS, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP):
    """
    返回:
      f: 频率轴 (Hz, 已中心化到 baseband, 需要时+FC还原真实频率)
      t: 时间轴 (s)
      Sxx_db: 功率谱 (dB), shape=(n_freq, n_time)
    """
    f, t, Zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                      return_onesided=False, boundary=None)
    f = np.fft.fftshift(f)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    Pxx = np.abs(Zxx) ** 2
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    return f, t, Pxx_db


def noise_floor_normalize(Sxx_db, pct=10):
    """
    对每个频率bin, 用时间维低百分位数估计噪底, 做相对dB归一化
    """
    noise_floor = np.percentile(Sxx_db, pct, axis=1, keepdims=True)  # (n_freq,1)
    Sxx_norm = Sxx_db - noise_floor
    return Sxx_norm, noise_floor.squeeze()


def preprocess(x):
    f, t, Sxx_db = compute_spectrogram(x)
    Sxx_norm, noise_floor = noise_floor_normalize(Sxx_db)
    return f, t, Sxx_norm, noise_floor


def extract_videoband_features(f, t, Sxx_norm, thresh_db=6.0, min_band_width_hz=1e5):
    """
    基于占用率提取图传子带结构特征
    """
    n_freq, n_time = Sxx_norm.shape
    active = Sxx_norm > thresh_db          # bool矩阵
    occ = active.mean(axis=1)              # 每个频点的时间占用率, shape=(n_freq,)

    # 用occupancy做频域连通域检测(简单阈值 + 连通标记)
    occ_active = occ > 0.15  # 该频点在>15%的时间片内活跃, 视为"图传占用带"候选
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
    feats['video_energy_ratio'] = float(np.sum(band_energy_ratio))  # 整体图传能量占比(区分background关键)
    feats['occ_profile_entropy'] = entropy(occ / (occ.sum() + 1e-12) + 1e-12)
    return feats


def _safe_skew(arr):
    from scipy.stats import skew
    return skew(arr) if len(arr) > 2 else 0.0


def extract_hopping_burst_features(f, t, Sxx_norm, video_band_mask=None,
                                    power_thresh_db=8.0, max_burst_dur=2e-3):
    """
    基于宽带瞬时功率检测短时突发(控制信号), 统计跳变特征
    """
    n_freq, n_time = Sxx_norm.shape
    if video_band_mask is None:
        video_band_mask = np.ones(n_freq, dtype=bool)

    # 宽带瞬时功率(线性域求和后转dB)
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
        if dur > max_burst_dur:   # 超过阈值判定为图传/长时占用, 非"控制信号"
            continue
        # 瞬时频率质心: 用该时间片功率谱加权
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
    """
    video_freq_range: (f_low, f_high) 图传主要占用区(可由特征组A统计得到的全局先验给出,
                       或直接用当前样本occupancy确定)
    """
    f_low, f_high = video_freq_range
    below_mask = f < f_low

    lin_power = 10 ** (Sxx_norm / 10)
    total_energy = lin_power.sum()
    below_energy = lin_power[below_mask, :].sum()

    active_below = (Sxx_norm[below_mask, :] > 6.0)
    active_ratio = active_below.mean() if active_below.size > 0 else 0.0

    # 全频带谱熵(background更接近白噪声, 熵更高更均匀)
    avg_psd = lin_power.mean(axis=1)
    p = avg_psd / (avg_psd.sum() + 1e-12)
    spec_entropy = entropy(p + 1e-12)

    feats = {
        'below_band_energy_ratio': below_energy / (total_energy + 1e-12),
        'below_band_active_ratio': active_ratio,
        'global_spectral_entropy': spec_entropy
    }
    return feats


def extract_statistical_features(x, f, avg_psd=None, Sxx_norm=None):
    if avg_psd is None:
        lin_power = 10 ** (Sxx_norm / 10)
        avg_psd = lin_power.mean(axis=1)

    p = avg_psd / (avg_psd.sum() + 1e-12)
    centroid = np.sum(f * p)
    bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * p))
    flatness = np.exp(np.mean(np.log(avg_psd + 1e-12))) / (np.mean(avg_psd) + 1e-12)
    spec_kurt = kurtosis(avg_psd)

    # 高阶累积量 (归一化功率的复基带信号)
    xn = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12)
    m20 = np.mean(xn ** 2)
    m21 = np.mean(xn * np.conj(xn))
    m40 = np.mean(xn ** 4)
    m41 = np.mean((xn ** 3) * np.conj(xn))
    m42 = np.mean((xn ** 2) * (np.conj(xn) ** 2))
    C40 = m40 - 3 * m20 ** 2
    C41 = m41 - 3 * m20 * m21
    C42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2

    feats = {
        'spec_centroid': centroid, 'spec_bandwidth': bandwidth,
        'spec_flatness': flatness, 'spec_kurtosis': spec_kurt,
        'C20_abs': np.abs(m20), 'C40_abs': np.abs(C40),
        'C41_abs': np.abs(C41), 'C42_abs': np.abs(C42)
    }
    return feats


def extract_all_features(fname, video_freq_range_global=(-2e7, 2e7)):
    x = load_iq(fname)
    f, t, Sxx_norm, noise_floor = preprocess(x)

    feats = {}
    feats.update(extract_videoband_features(f, t, Sxx_norm))
    feats.update(extract_hopping_burst_features(f, t, Sxx_norm))
    feats.update(extract_interference_features(f, t, Sxx_norm, video_freq_range_global))
    feats.update(extract_statistical_features(x, f, Sxx_norm=Sxx_norm))
    return feats


from tqdm import tqdm
import time
import traceback

def build_feature_table(df_meta, save_path='feat_partial.csv', save_every=50):
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
            print(f"已处理 {i+1}/{len(df_meta)}, 累计耗时 {time.time()-t0:.1f}s, 已保存中间结果")

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"全部完成, 总耗时 {time.time()-t0:.1f}s")
    return df

# ===== 类别标签统一改为文件名首字母(大写); Background -> 'B' =====
KNOWN_CLASSES = ['A', 'D', 'E', 'G']
UNKNOWN_CLASSES = ['B', 'C', 'F']


def prepare_known_unknown(feat_df, feature_cols):
    known_df = feat_df[feat_df['mtype'].isin(KNOWN_CLASSES)].copy()
    unknown_df = feat_df[feat_df['mtype'].isin(UNKNOWN_CLASSES)].copy()

    scaler = StandardScaler()
    scaler.fit(known_df[feature_cols].values)  # 仅在已知类上拟合, 防止信息泄露

    known_df[feature_cols] = scaler.transform(known_df[feature_cols].values)
    unknown_df[feature_cols] = scaler.transform(unknown_df[feature_cols].values)
    return known_df, unknown_df, scaler


def intra_class_compactness(known_df, feature_cols):
    results = {}
    X = known_df[feature_cols].values
    for c in KNOWN_CLASSES:
        Xc = known_df.loc[known_df['mtype'] == c, feature_cols].values
        mu = Xc.mean(axis=0)
        d2 = np.sum((Xc - mu) ** 2, axis=1)
        results[c] = {
            'mean_dist2': d2.mean(), 'std_dist2': d2.std(),
            'n_samples': len(Xc)
        }
    S_W = np.mean([v['mean_dist2'] for v in results.values()])
    return results, S_W


def inter_class_separation(known_df, feature_cols):
    X = known_df[feature_cols].values
    y = known_df['mtype'].values

    mu_global = X.mean(axis=0)
    S_B = 0.0
    for c in KNOWN_CLASSES:
        Xc = known_df.loc[known_df['mtype'] == c, feature_cols].values
        mu_c = Xc.mean(axis=0)
        S_B += len(Xc) * np.sum((mu_c - mu_global) ** 2)

    _, S_W = intra_class_compactness(known_df, feature_cols)
    fisher_J = S_B / (S_W + 1e-12)

    sil = silhouette_score(X, y)
    ch = calinski_harabasz_score(X, y)
    db = davies_bouldin_score(X, y)

    # 已知类中心两两欧式距离
    centers = {c: known_df.loc[known_df['mtype'] == c, feature_cols].values.mean(axis=0)
               for c in KNOWN_CLASSES}
    center_dist = pd.DataFrame(
        [[np.linalg.norm(centers[a] - centers[b]) for b in KNOWN_CLASSES] for a in KNOWN_CLASSES],
        index=KNOWN_CLASSES, columns=KNOWN_CLASSES)

    return {'fisher_J': fisher_J, 'silhouette': sil,
            'calinski_harabasz': ch, 'davies_bouldin': db,
            'center_dist_matrix': center_dist}


def rbf_mmd2(X, Y, gamma=None):
    """简单RBF核MMD^2估计, 用于比较IN/OUT分布差异"""
    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        # median heuristic
        Z = np.vstack([X, Y])
        from scipy.spatial.distance import pdist
        gamma = 1.0 / (2 * np.median(pdist(Z)) ** 2 + 1e-12)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def stability_individual(known_df, feature_cols):
    """个体级Fisher比: 个体间方差 vs 机型间方差"""
    results = {}
    for c in KNOWN_CLASSES:
        sub = known_df[known_df['mtype'] == c]
        mu_c = sub[feature_cols].values.mean(axis=0)
        S_W_indiv = 0.0
        for indiv in sub['indiv'].unique():
            Xi = sub.loc[sub['indiv'] == indiv, feature_cols].values
            mu_i = Xi.mean(axis=0)
            S_W_indiv += np.sum((Xi - mu_i) ** 2)
        S_B_indiv = 0.0
        for indiv in sub['indiv'].unique():
            Xi = sub.loc[sub['indiv'] == indiv, feature_cols].values
            mu_i = Xi.mean(axis=0)
            S_B_indiv += len(Xi) * np.sum((mu_i - mu_c) ** 2)
        results[c] = S_B_indiv / (S_W_indiv + 1e-12)
    return results  # 值越小, 说明个体差异对特征影响越小(稳定性好)


def stability_condition(known_df, feature_cols):
    """IN vs OUT 分布差异 (MMD)"""
    results = {}
    for c in KNOWN_CLASSES:
        sub = known_df[known_df['mtype'] == c]
        X_in = sub.loc[sub['cond'] == 'IN', feature_cols].values
        X_out = sub.loc[sub['cond'] == 'OUT', feature_cols].values
        if len(X_in) > 1 and len(X_out) > 1:
            results[c] = rbf_mmd2(X_in, X_out)
        else:
            results[c] = np.nan
    return results


def stability_time(known_df, feature_cols):
    """时间序号(S0,S1,...)间特征漂移: 用组间/组内方差比"""
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
    """
    类别可解释方差占比 (简化版单向随机效应ICC, 对特征向量取整体范数或PC1近似)
    这里用每一维特征分别算ICC后取均值, 更严谨可换成MANOVA
    """
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


def unknown_distance_analysis(known_df, unknown_df, feature_cols):
    centers = {c: known_df.loc[known_df['mtype'] == c, feature_cols].values.mean(axis=0)
               for c in KNOWN_CLASSES}
    cov_pooled = LedoitWolf().fit(known_df[feature_cols].values).covariance_
    cov_inv = np.linalg.pinv(cov_pooled)

    def maha_min(x):
        dists = [np.sqrt((x - mu) @ cov_inv @ (x - mu)) for mu in centers.values()]
        return min(dists)

    # known类内(留出)距离作为参照分布: 简单用留一法/或直接用训练集类内距离
    known_intra_dist = []
    for c in KNOWN_CLASSES:
        Xc = known_df.loc[known_df['mtype'] == c, feature_cols].values
        mu = centers[c]
        for x in Xc:
            known_intra_dist.append(np.sqrt((x - mu) @ cov_inv @ (x - mu)))
    known_intra_dist = np.array(known_intra_dist)

    result = {}
    for uc in UNKNOWN_CLASSES:
        Xu = unknown_df.loc[unknown_df['mtype'] == uc, feature_cols].values
        dists = np.array([maha_min(x) for x in Xu])
        ks_stat, ks_p = ks_2samp(known_intra_dist, dists)
        result[uc] = {
            'mean_dist': dists.mean(), 'std_dist': dists.std(),
            'ks_stat_vs_known_intra': ks_stat, 'ks_pvalue': ks_p,
            'raw_dists': dists
        }
    result['known_intra_dist'] = known_intra_dist
    return result


def known_unknown_separability(known_df, unknown_df, feature_cols, fpr_target=0.05):
    centers = {c: known_df.loc[known_df['mtype'] == c, feature_cols].values.mean(axis=0)
               for c in KNOWN_CLASSES}
    cov_pooled = LedoitWolf().fit(known_df[feature_cols].values).covariance_
    cov_inv = np.linalg.pinv(cov_pooled)

    def maha_min(x):
        dists = [np.sqrt((x - mu) @ cov_inv @ (x - mu)) for mu in centers.values()]
        return min(dists)

    # 用known的一部分做"负样本"(留出), 简单起见这里用全部known(实践中应做train/val切分)
    X_known = known_df[feature_cols].values
    score_known = np.array([maha_min(x) for x in X_known])
    y_known = np.zeros(len(score_known))  # 0=known

    results = {}
    for uc in UNKNOWN_CLASSES:
        Xu = unknown_df.loc[unknown_df['mtype'] == uc, feature_cols].values
        score_u = np.array([maha_min(x) for x in Xu])
        y_u = np.ones(len(score_u))  # 1=unknown

        y_all = np.concatenate([y_known, y_u])
        score_all = np.concatenate([score_known, score_u])

        auroc = roc_auc_score(y_all, score_all)
        aupr = average_precision_score(y_all, score_all)

        fpr, tpr, thr = roc_curve(y_all, score_all)
        idx = np.argmin(np.abs(fpr - fpr_target))
        tpr_at_fpr = tpr[idx]

        results[uc] = {'AUROC': auroc, 'AUPR': aupr,
                        f'TPR@FPR={fpr_target}': tpr_at_fpr,
                        'roc_curve': (fpr, tpr)}

    # 汇总: 三个unknown合并
    Xu_all = unknown_df[feature_cols].values
    score_u_all = np.array([maha_min(x) for x in Xu_all])
    y_all = np.concatenate([y_known, np.ones(len(score_u_all))])
    score_all = np.concatenate([score_known, score_u_all])
    results['overall'] = {
        'AUROC': roc_auc_score(y_all, score_all),
        'AUPR': average_precision_score(y_all, score_all)
    }
    return results


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
        # perplexity 必须小于样本数, 且一般建议在 5~50 之间
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


if __name__ == '__main__':
    ROOT = 'E:/XieYiR/UAV_osr_X/twin_droneRFa/twin_droneRF/train'
    FEAT_CSV = 'feat_all.csv'

    df_meta = scan_dataset(ROOT)
    print(f"共扫描到 {len(df_meta)} 个样本, 类别分布:\n{df_meta['mtype'].value_counts()}")

    if os.path.exists(FEAT_CSV):
        print(f"检测到已有特征文件 {FEAT_CSV}, 直接加载")
        feat_df = pd.read_csv(FEAT_CSV)
    else:
        feat_df = build_feature_table(df_meta, save_path='feat_partial.csv')
        feat_df.to_csv(FEAT_CSV, index=False)
    feature_cols = [c for c in feat_df.columns
                    if c not in ['mtype', 'indiv', 'cond', 'seq', 'slice', 'file']]

    known_df, unknown_df, scaler = prepare_known_unknown(feat_df, feature_cols)

    # 1. 类内紧凑性
    intra_res, S_W = intra_class_compactness(known_df, feature_cols)
    print("\n[类内紧凑性]", intra_res, "\nS_W =", S_W)

    # 2. 类间分离度
    inter_res = inter_class_separation(known_df, feature_cols)
    print("\n[类间分离度]", {k: v for k, v in inter_res.items() if k != 'center_dist_matrix'})
    print(inter_res['center_dist_matrix'])

    # 3. 特征稳定性
    print("\n[个体稳定性 Fisher比, 越小越好]", stability_individual(known_df, feature_cols))
    print("[IN/OUT MMD^2, 越小越好]", stability_condition(known_df, feature_cols))
    print("[时间序号稳定性 比值, 越小越好]", stability_time(known_df, feature_cols))
    mean_icc, icc_detail = overall_ICC(known_df, feature_cols)
    print("[整体类别ICC, 越接近1越好]", mean_icc)

    # 4. 未知类距离
    dist_res = unknown_distance_analysis(known_df, unknown_df, feature_cols)
    for uc in UNKNOWN_CLASSES:
        print(f"\n[未知类距离] {uc}: mean={dist_res[uc]['mean_dist']:.3f}, "
              f"KS={dist_res[uc]['ks_stat_vs_known_intra']:.3f}, p={dist_res[uc]['ks_pvalue']:.4f}")

    # 5. Known/Unknown可分性
    sep_res = known_unknown_separability(known_df, unknown_df, feature_cols)
    for uc in UNKNOWN_CLASSES + ['overall']:
        print(f"\n[K/U可分性] {uc}: AUROC={sep_res[uc]['AUROC']:.3f}, "
              f"AUPR={sep_res[uc]['AUPR']:.3f}")

    # 可视化
    visualize_feature_space(known_df, unknown_df, feature_cols, method='tsne')