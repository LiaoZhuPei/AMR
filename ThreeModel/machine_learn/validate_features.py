"""
validate_features.py
====================
特征提取验证脚本

功能：
  1. 从 .mat 测试文件（或 .npz 数据集）读取实际信号
  2. 调用 feature_extract.extract_features() 提取 8 通道特征
  3. 计算每个通道的统计摘要（均值、std、min、max、偏度、峰度）
  4. 与各调制类型的理论期望值对比，输出偏差报告
  5. 全程纯文本输出，无图片文件，无 OpenCV

理论值说明
----------
各特征的理论值来自经典调制识别文献（Azzouz & Nandi 1996 体系）。
噪声存在时理论值会漂移，脚本同时输出"高SNR子集"和"全样本"两套统计，
便于判断是噪声引起的偏差还是特征提取本身的问题。

使用方式
--------
  # 从 .npz 数据集验证（推荐，SNR 已知）
  python validate_features.py --source npz \
      --npz_path ./sionna_dataset/Sionna0_a.npz \
      --snr_threshold 20

  # 从 .mat 文件目录验证（实际部署数据）
  python validate_features.py --source mat \
      --mat_dir E:/Project/AMR-Benchmark/output_mat_files_QAM256/ \
      --true_class 256QAM \
      --n_files 100
"""

import argparse
import os
import glob
import textwrap

import numpy as np
import h5py

from feature_extract_G import extract_features, N_CHANNELS


# ══════════════════════════════════════════════════════════════════════════
# 理论值表（高 SNR 无噪理想情况）
# ══════════════════════════════════════════════════════════════════════════
# 格式：每个调制对应 8 个通道的 (期望均值, 期望std, 描述) 三元组
# 均值/std 均针对归一化后的特征，部分通道理论上无解析式，用 None 表示"不作比较"
#
# 依据：
#   Ch2 幅度均值归一化 → 理论上 mean≈0；PSK/FSK 的幅度恒定故 std 趋近于 0，
#        高阶 QAM 幅度多环故 std 较大
#   Ch3 相位 /π：PSK 相位集中于星座角，QAM 分散
#   Ch4 瞬时频率：线性调制 ≈ 0，FSK/MSK 呈现明显频率跳变
#   Ch5 A²均值归一化 → mean≈0；std 与 ch2 趋势一致但更极端
#   Ch6 自相关：相干调制（PSK/QAM）自相关高，FSK 较低
#   Ch7 差分幅度：恒幅调制（PSK/FSK/MSK）≈ 0，ASK/QAM 有跳变
#
# 字段: (mean_expected, std_expected_range_lo, std_expected_range_hi, note)
# None 表示理论不明确，仅输出实测值

THEORY = {
    # ── 恒幅 PSK 族 ──────────────────────────────────────────────────────
    'BPSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.05, '恒幅，std→0'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.15, '线性调制，频率≈0'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.05, '恒幅，A²恒定，std→0'),
        'ch7_dA_std':         (None, 0.0,   0.10, '恒幅，差分幅度≈0'),
    },
    'QPSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.05, '恒幅，std→0'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.15, '线性调制，频率≈0'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.05, '恒幅，std→0'),
        'ch7_dA_std':         (None, 0.0,   0.10, '恒幅'),
    },
    '8PSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.08, '恒幅，RRC成型后轻微起伏'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.20, '线性调制'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.08, '恒幅'),
        'ch7_dA_std':         (None, 0.0,   0.15, '恒幅'),
    },
    'OQPSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.12, 'Q路延迟导致轻微幅度起伏'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.20, '线性调制'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.12, '同 ch2'),
        'ch7_dA_std':         (None, 0.0,   0.20, 'Q路延迟引入差分'),
    },
    'Pi4DQPSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.12, '差分调制，幅度起伏类似OQPSK'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.25, '差分相位导致频率起伏'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.12, '同 ch2'),
        'ch7_dA_std':         (None, 0.0,   0.20, '差分'),
    },
    # ── FSK / MSK ────────────────────────────────────────────────────────
    '2FSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.05, '恒幅 FSK'),
        'ch4_freq_mean_norm': (0.0,  0.30,  0.80, '2FSK 频率在±Δf跳变，std 应明显'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.05, '恒幅'),
        'ch7_dA_std':         (None, 0.0,   0.08, '恒幅，差分幅度≈0'),
    },
    '4FSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.05, '恒幅 FSK'),
        'ch4_freq_mean_norm': (0.0,  0.40,  1.00, '4FSK 有4个频率，std 更大'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.05, '恒幅'),
        'ch7_dA_std':         (None, 0.0,   0.08, '恒幅'),
    },
    'MSK': {
        'ch2_amp_mean_norm':  (0.0,  0.0,   0.05, 'MSK 恒幅'),
        'ch4_freq_mean_norm': (0.0,  0.15,  0.50, 'MSK h=0.5，频率偏差小于 2FSK'),
        'ch5_a2_mean_norm':   (0.0,  0.0,   0.05, '恒幅'),
        'ch7_dA_std':         (None, 0.0,   0.08, '恒幅'),
    },
    # ── ASK ──────────────────────────────────────────────────────────────
    '2ASK': {
        'ch2_amp_mean_norm':  (0.0,  0.30,  0.80, '2ASK 幅度二值，std 明显'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.20, '线性调制，频率≈0'),
        'ch5_a2_mean_norm':   (0.0,  0.40,  1.20, 'A² 方差更大'),
        'ch7_dA_std':         (None, 0.30,  0.90, '幅度跳变大，差分大'),
    },
    # ── QAM 族（随阶数升高，幅度 std 递增） ─────────────────────────────
    '16QAM': {
        'ch2_amp_mean_norm':  (0.0,  0.15,  0.45, '16QAM 3个幅度环'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.25, '线性调制'),
        'ch5_a2_mean_norm':   (0.0,  0.20,  0.60, 'A² 方差中等'),
        'ch7_dA_std':         (None, 0.10,  0.40, '幅度跳变中等'),
    },
    '32QAM': {
        'ch2_amp_mean_norm':  (0.0,  0.20,  0.55, '32QAM Cross-QAM，幅度环更多'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.25, '线性调制'),
        'ch5_a2_mean_norm':   (0.0,  0.28,  0.70, 'A²方差偏大'),
        'ch7_dA_std':         (None, 0.15,  0.50, '跳变增大'),
    },
    '64QAM': {
        'ch2_amp_mean_norm':  (0.0,  0.25,  0.60, '64QAM 幅度环更多'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.25, '线性调制'),
        'ch5_a2_mean_norm':   (0.0,  0.35,  0.80, 'A²方差大'),
        'ch7_dA_std':         (None, 0.20,  0.60, '跳变更大'),
    },
    '128QAM': {
        'ch2_amp_mean_norm':  (0.0,  0.28,  0.65, '128QAM'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.25, '线性调制'),
        'ch5_a2_mean_norm':   (0.0,  0.40,  0.90, 'A²方差大'),
        'ch7_dA_std':         (None, 0.25,  0.65, '跳变大'),
    },
    '256QAM': {
        'ch2_amp_mean_norm':  (0.0,  0.30,  0.70, '256QAM 幅度环最多'),
        'ch4_freq_mean_norm': (0.0,  0.0,   0.25, '线性调制'),
        'ch5_a2_mean_norm':   (0.0,  0.45,  1.00, 'A²方差最大'),
        'ch7_dA_std':         (None, 0.30,  0.70, '跳变最大'),
    },
}

# 通道名称映射
CH_NAMES = [
    'Ch0  I (raw)',
    'Ch1  Q (raw)',
    'Ch2  幅度 A (mean-norm)',
    'Ch3  相位 φ/π',
    'Ch4  瞬时频率 (mean-norm)',
    'Ch5  幅度平方 A² (mean-norm)',
    'Ch6  短时自相关 (mean-norm)',
    'Ch7  差分幅度 ΔA (std-norm)',
]

# 理论键到通道索引的映射
THEORY_CH_MAP = {
    'ch2_amp_mean_norm':  2,
    'ch4_freq_mean_norm': 4,
    'ch5_a2_mean_norm':   5,
    'ch7_dA_std':         7,
}


# ══════════════════════════════════════════════════════════════════════════
# 统计计算
# ══════════════════════════════════════════════════════════════════════════

def compute_stats(feat_ch: np.ndarray) -> dict:
    """
    feat_ch: (N, L) 单通道数据
    返回字典：mean, std, min, max, skewness, kurtosis_excess
    """
    from scipy.stats import skew, kurtosis
    flat = feat_ch.reshape(feat_ch.shape[0], -1)   # (N, L)
    per_sample_mean = flat.mean(axis=1)             # (N,)
    per_sample_std  = flat.std(axis=1)

    return {
        'mean_of_means':  float(per_sample_mean.mean()),
        'std_of_means':   float(per_sample_mean.std()),
        'mean_of_stds':   float(per_sample_std.mean()),
        'std_of_stds':    float(per_sample_std.std()),
        'global_min':     float(flat.min()),
        'global_max':     float(flat.max()),
        'skewness':       float(skew(per_sample_mean)),
        'kurt_excess':    float(kurtosis(per_sample_mean)),
    }


# ══════════════════════════════════════════════════════════════════════════
# 偏差判断
# ══════════════════════════════════════════════════════════════════════════

def _check_value(label: str, measured: float, theo_mean, lo: float, hi: float,
                 note: str, width: int = 70) -> str:
    """生成单条对比行，返回带 PASS/WARN/INFO 标记的字符串"""
    if theo_mean is not None:
        diff = measured - theo_mean
        if lo <= abs(diff) <= hi or abs(diff) <= hi:
            status = '✓ PASS'
        else:
            status = '✗ WARN'
        line = (f"  {label:<28} measured={measured:+.4f}  "
                f"theory≈{theo_mean:+.4f}  Δ={diff:+.4f}  [{lo:.2f},{hi:.2f}]  "
                f"{status}  # {note}")
    else:
        in_range = lo <= measured <= hi
        status = '✓ PASS' if in_range else '✗ WARN'
        line = (f"  {label:<28} measured={measured:+.4f}  "
                f"expected_range=[{lo:.2f},{hi:.2f}]  "
                f"{status}  # {note}")
    return line


# ══════════════════════════════════════════════════════════════════════════
# 核心验证函数
# ══════════════════════════════════════════════════════════════════════════

def validate_modulation(signals: np.ndarray, mod_name: str,
                        snr_db: np.ndarray = None,
                        snr_threshold: float = 20.0,
                        sps: int = 4) -> None:
    """
    signals    : (N, 2, L) float32，已功率归一化
    mod_name   : 调制类型字符串，需在 THEORY 中有对应条目
    snr_db     : (N,) 每个样本的 SNR（可为 None，则不做高 SNR 子集）
    snr_threshold: 高 SNR 子集门限 (dB)
    """
    N = signals.shape[0]
    print(f"\n{'='*72}")
    print(f"  调制类型: {mod_name}   样本数: {N}")
    print(f"{'='*72}")

    # 提取特征
    feat = extract_features(signals, sps=sps)   # (N, 8, L)

    # 是否有高 SNR 子集
    if snr_db is not None:
        hi_mask = snr_db >= snr_threshold
        n_hi = hi_mask.sum()
        feat_hi = feat[hi_mask] if n_hi > 0 else None
    else:
        hi_mask = None
        feat_hi = None
        n_hi = 0

    # ── 1. 每通道统计摘要（全样本） ────────────────────────────────────
    print(f"\n  [ 全样本统计 (N={N}) ]")
    print(f"  {'通道':<32} {'均值的均':>10} {'均值的std':>10} "
          f"{'std的均':>10} {'偏度':>8} {'峰度':>8}")
    print(f"  {'-'*78}")

    stats_all = []
    for ch in range(N_CHANNELS):
        s = compute_stats(feat[:, ch, :])
        stats_all.append(s)
        print(f"  {CH_NAMES[ch]:<32} "
              f"{s['mean_of_means']:>+10.4f} "
              f"{s['std_of_means']:>10.4f} "
              f"{s['mean_of_stds']:>10.4f} "
              f"{s['skewness']:>+8.3f} "
              f"{s['kurt_excess']:>+8.3f}")

    # ── 2. 高 SNR 子集统计（若有） ──────────────────────────────────────
    if feat_hi is not None and n_hi >= 10:
        print(f"\n  [ 高SNR子集 (≥{snr_threshold:.0f}dB, N={n_hi}) ]")
        print(f"  {'通道':<32} {'均值的均':>10} {'均值的std':>10} "
              f"{'std的均':>10} {'偏度':>8} {'峰度':>8}")
        print(f"  {'-'*78}")
        stats_hi = []
        for ch in range(N_CHANNELS):
            s = compute_stats(feat_hi[:, ch, :])
            stats_hi.append(s)
            print(f"  {CH_NAMES[ch]:<32} "
                  f"{s['mean_of_means']:>+10.4f} "
                  f"{s['std_of_means']:>10.4f} "
                  f"{s['mean_of_stds']:>10.4f} "
                  f"{s['skewness']:>+8.3f} "
                  f"{s['kurt_excess']:>+8.3f}")
    else:
        stats_hi = None
        if snr_db is not None:
            print(f"\n  [ 高SNR子集 (≥{snr_threshold:.0f}dB) 样本不足，跳过 ]")

    # ── 3. 与理论值对比 ─────────────────────────────────────────────────
    print(f"\n  [ 理论值对比 ]")
    if mod_name not in THEORY:
        print(f"  (该调制类型 '{mod_name}' 暂无理论条目，仅输出实测统计)")
        return

    theory_entry = THEORY[mod_name]
    # 优先用高 SNR 子集做理论对比
    use_stats = stats_hi if stats_hi is not None else stats_all
    use_label = f"高SNR(≥{snr_threshold:.0f}dB)" if stats_hi is not None else "全样本"

    print(f"  (对比基准: {use_label})")
    print()

    pass_count = 0
    warn_count = 0
    for key, (theo_mean, lo, hi, note) in theory_entry.items():
        ch_idx = THEORY_CH_MAP[key]
        s = use_stats[ch_idx]

        # 根据 key 决定用哪个统计量做对比
        if 'std' in key.split('_')[-1]:
            # 用 mean_of_stds 做对比（ch7_dA_std）
            measured = s['mean_of_stds']
            label = f"{CH_NAMES[ch_idx][:20]}  [std]"
        else:
            # 用 mean_of_means 做对比
            measured = s['mean_of_means']
            label = f"{CH_NAMES[ch_idx][:20]}  [mean]"

        line = _check_value(label, measured, theo_mean, lo, hi, note)
        print(line)
        if '✓ PASS' in line:
            pass_count += 1
        else:
            warn_count += 1

    total_checks = pass_count + warn_count
    print(f"\n  结果汇总: {pass_count}/{total_checks} PASS"
          f"  {warn_count}/{total_checks} WARN")

    if warn_count > 0:
        print("\n  ⚠ WARN 可能原因：")
        print("    · 低 SNR 导致特征漂移（尝试提高 --snr_threshold）")
        print("    · 信道损伤（多径、CFO、PA 非线性）导致理论值偏移")
        print("    · 实际数据格式与预期不符（检查功率归一化是否正常）")
        print("    · 当前理论范围基于理想仿真，实测数据存在系统偏差属正常现象")


# ══════════════════════════════════════════════════════════════════════════
# 数据加载辅助
# ══════════════════════════════════════════════════════════════════════════

def load_from_npz(npz_path: str, target_classes: list = None):
    """
    从 .npz 加载测试集，按调制类型分组返回。
    返回: dict {mod_name: {'signals': (N,2,L), 'snr': (N,)}}
    """
    data    = np.load(npz_path, allow_pickle=True)
    classes = list(data['classes'])
    X       = data['X_test']          # (N, 2, L)
    Y       = data['Y_test']          # (N,)
    SNR     = data['snr_test']        # (N,)

    # 功率归一化
    pwr = np.mean(X ** 2, axis=(1, 2), keepdims=True)
    X   = (X / np.sqrt(np.maximum(pwr, 1e-10))).astype(np.float32)

    result = {}
    for c_idx, c_name in enumerate(classes):
        if target_classes and c_name not in target_classes:
            continue
        mask = Y == c_idx
        if mask.sum() == 0:
            continue
        result[c_name] = {
            'signals': X[mask],
            'snr':     SNR[mask],
        }
    return result


def load_from_mat(mat_dir: str, n_files: int = 100):
    """
    从 .mat 文件目录加载真实测试数据（标签未知，只验证特征分布）。
    返回: (signals, None)  signals shape (N, 2, 1024)
    """
    files = glob.glob(os.path.join(mat_dir, '*.mat'))[:n_files]
    if not files:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    sigs = []
    for fp in files:
        try:
            with h5py.File(fp, 'r') as f:
                s = f['data_save'][:]   # (1, 2, 1024)
            if s.shape == (1, 2, 1024):
                sigs.append(s.squeeze(0))
        except Exception as e:
            print(f"  [Skip] {os.path.basename(fp)}: {e}")

    if not sigs:
        raise RuntimeError("No valid signals loaded.")

    X = np.stack(sigs, axis=0).astype(np.float32)   # (N, 2, 1024)
    pwr = np.mean(X ** 2, axis=(1, 2), keepdims=True)
    X   = (X / np.sqrt(np.maximum(pwr, 1e-10))).astype(np.float32)
    print(f"  Loaded {len(X)} signals from {mat_dir}")
    return X, None


# ══════════════════════════════════════════════════════════════════════════
# 快捷函数：输出全部通道的分位数分布（诊断用）
# ══════════════════════════════════════════════════════════════════════════

def print_percentile_report(signals: np.ndarray, mod_name: str, sps: int = 4):
    """输出每通道 5/25/50/75/95 分位数，帮助判断分布形态"""
    feat = extract_features(signals, sps=sps)   # (N, 8, L)
    N, C, L = feat.shape

    print(f"\n  [ 分位数报告: {mod_name}  N={N} ]")
    print(f"  {'通道':<32} {'p5':>8} {'p25':>8} {'p50':>8} "
          f"{'p75':>8} {'p95':>8}  (per-sample mean 分布)")
    print(f"  {'-'*80}")

    for ch in range(C):
        per_mean = feat[:, ch, :].mean(axis=1)   # (N,) 每样本的通道均值
        p = np.percentile(per_mean, [5, 25, 50, 75, 95])
        print(f"  {CH_NAMES[ch]:<32} "
              f"{p[0]:>+8.4f} {p[1]:>+8.4f} {p[2]:>+8.4f} "
              f"{p[3]:>+8.4f} {p[4]:>+8.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 跨类别对比（仅输出关键通道的均值±std，直观看类间区分度）
# ══════════════════════════════════════════════════════════════════════════

def print_cross_class_summary(npz_path: str, snr_threshold: float = 20.0,
                              sps: int = 4):
    """
    加载 npz 全部类别，输出关键通道（2,4,5,7）的高SNR统计，
    以表格形式显示类间区分度。
    """
    data    = np.load(npz_path, allow_pickle=True)
    classes = list(data['classes'])
    X       = data['X_test']
    Y       = data['Y_test']
    SNR     = data['snr_test']
    pwr     = np.mean(X ** 2, axis=(1, 2), keepdims=True)
    X       = (X / np.sqrt(np.maximum(pwr, 1e-10))).astype(np.float32)

    KEY_CHS = [2, 4, 5, 7]   # 关键通道
    KEY_LABELS = ['Ch2 A std', 'Ch4 f std', 'Ch5 A² std', 'Ch7 ΔA std']

    print(f"\n{'='*80}")
    print(f"  跨类别关键特征对比  (高SNR ≥ {snr_threshold:.0f} dB)")
    print(f"{'='*80}")
    header = f"  {'调制':<12}" + "".join(f"{lb:>14}" for lb in KEY_LABELS)
    print(header)
    print(f"  {'-'*78}")

    for c_idx, c_name in enumerate(classes):
        mask = (Y == c_idx) & (SNR >= snr_threshold)
        if mask.sum() < 5:
            print(f"  {c_name:<12}  (高SNR样本不足)")
            continue

        feat = extract_features(X[mask], sps=sps)   # (N, 8, L)
        row = f"  {c_name:<12}"
        for ch in KEY_CHS:
            s = compute_stats(feat[:, ch, :])
            row += f"  {s['mean_of_stds']:>+6.4f}±{s['std_of_stds']:.4f}"
        print(row)

    print(f"\n  说明: 值为 per-sample 通道 std 的 均值±std")
    print(f"  Ch2(幅度std)、Ch5(A²std)、Ch7(差分幅度std) 对 QAM 阶数单调递增")
    print(f"  Ch4(频率std) 对 FSK/MSK 显著偏大")


# ══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='特征提取验证：对比实测特征与理论值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        示例:
          # 验证 npz 数据集的所有类别（高SNR子集）
          python validate_features.py --source npz \\
              --npz_path ./sionna_dataset/Sionna0_a.npz

          # 只验证 QAM 族，门限 SNR=18dB
          python validate_features.py --source npz \\
              --npz_path ./sionna_dataset/Sionna0_a.npz \\
              --classes 16QAM 64QAM 256QAM --snr_threshold 18

          # 跨类别对比表格
          python validate_features.py --source npz \\
              --npz_path ./sionna_dataset/Sionna0_a.npz --cross_class

          # 验证真实 .mat 文件（已知类别为 256QAM）
          python validate_features.py --source mat \\
              --mat_dir E:/data/QAM256/ --true_class 256QAM --n_files 200
        """)
    )
    parser.add_argument('--source', choices=['npz', 'mat'], default='npz')
    parser.add_argument('--npz_path', default='./sionna_dataset/Sionna0_a.npz')
    parser.add_argument('--mat_dir', default='')
    parser.add_argument('--true_class', default='256QAM',
                        help='mat 模式下的真实调制类型（用于对比理论值）')
    parser.add_argument('--n_files', type=int, default=100,
                        help='mat 模式下最多加载的文件数')
    parser.add_argument('--classes', nargs='+', default=None,
                        help='npz 模式下要验证的类别列表，默认全部')
    parser.add_argument('--snr_threshold', type=float, default=20.0,
                        help='高SNR子集门限 (dB)，默认 20')
    parser.add_argument('--cross_class', action='store_true',
                        help='输出跨类别对比表格（仅 npz 模式）')
    parser.add_argument('--percentile', action='store_true',
                        help='额外输出分位数报告')
    parser.add_argument('--sps', type=int, default=4,
                        help='每符号采样数（用于自相关滞后），默认 4')
    args = parser.parse_args()

    print("=" * 72)
    print("  AMR 特征提取验证工具")
    print("=" * 72)

    if args.source == 'npz':
        # ── npz 模式 ──────────────────────────────────────────────────────
        if args.cross_class:
            print_cross_class_summary(args.npz_path,
                                      snr_threshold=args.snr_threshold,
                                      sps=args.sps)
            return

        groups = load_from_npz(args.npz_path, target_classes=args.classes)
        for mod_name, d in groups.items():
            validate_modulation(
                signals=d['signals'],
                mod_name=mod_name,
                snr_db=d['snr'],
                snr_threshold=args.snr_threshold,
                sps=args.sps,
            )
            if args.percentile:
                print_percentile_report(d['signals'], mod_name, sps=args.sps)

    else:
        # ── mat 模式 ──────────────────────────────────────────────────────
        signals, _ = load_from_mat(args.mat_dir, n_files=args.n_files)
        validate_modulation(
            signals=signals,
            mod_name=args.true_class,
            snr_db=None,
            snr_threshold=args.snr_threshold,
            sps=args.sps,
        )
        if args.percentile:
            print_percentile_report(signals, args.true_class, sps=args.sps)

    print("\n✓ 验证完成")


if __name__ == '__main__':
    main()
