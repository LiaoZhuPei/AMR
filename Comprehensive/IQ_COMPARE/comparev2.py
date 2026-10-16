"""
仿真 vs 实采 IQ 数据对比分析
用法: python iq_compare.py --sim sim_128QAM.npz --real ./output_mat_files_128QAM --outdir ./results
"""

import os, sys, argparse, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import h5py
from tqdm import tqdm

# ── 配置 Matplotlib 解决中文乱码与负号显示问题 ───────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')
C_SIM  = '#1D9E75'
C_REAL = '#185FA5'
C_WARN = '#BA7517'
C_BAD  = '#A32D2D'
C_OK   = '#3B6D11'


# ╔══════════════════╗
# ║  1. 数据加载与处理 ║
# ╚══════════════════╝

def load_sim(path: str) -> tuple[np.ndarray, str, float]:
    """加载 gen_single_mod.py 生成的 .npz，返回 (X, mod, snr)"""
    d = np.load(path, allow_pickle=True)
    X   = d['X']                        # (N, 2, 1024)
    mod = str(d['mod'])
    snr = float(d['snr'])
    print(f'[仿真] {mod}  SNR={snr:+.0f}dB  样本数={len(X):,}  形状={X.shape}')
    return X, mod, snr


def load_real(path: str, max_n: int = 2000) -> np.ndarray:
    """
    加载实采数据目录或单文件，统一返回 (N, 2, 1024) float32
    支持 .mat (HDF5) / .npy / .npz
    """
    p = Path(path)
    files = []
    if p.is_dir():
        for ext in ('*.mat', '*.npy', '*.npz'):
            files += sorted(p.glob(ext))
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(f'路径不存在: {path}')

    segs = []
    for f in tqdm(files, desc='[实采] 加载', ncols=80):
        if len(segs) >= max_n:
            break
        try:
            segs.append(_load_one(f))
        except Exception as e:
            print(f'  ⚠ 跳过 {f.name}: {e}')

    if not segs:
        raise RuntimeError('未能加载任何实采片段，请检查路径和文件格式')

    X = np.concatenate(segs, axis=0)[:max_n]
    print(f'[实采] 片段总数={len(X):,}  形状={X.shape}')
    return X.astype(np.float32)


def _load_one(path: Path) -> np.ndarray:
    """单文件 → (N, 2, 1024)"""
    suf = path.suffix.lower()
    if suf == '.mat':
        return _load_mat(path)
    elif suf == '.npy':
        return _arr_to_iq(np.load(path))
    elif suf == '.npz':
        d = np.load(path, allow_pickle=True)
        key = 'data_save' if 'data_save' in d else d.files[0]
        return _arr_to_iq(d[key])
    raise ValueError(f'不支持的格式: {suf}')


def _load_mat(path: Path) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        key = 'data_save' if 'data_save' in f else list(f.keys())[0]
        arr = f[key][()]
    return _arr_to_iq(arr)


def _arr_to_iq(arr: np.ndarray) -> np.ndarray:
    """将各种形状转为 (N, 2, 1024)"""
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim == 2:
        if arr.shape[0] == 2:       return arr[np.newaxis]       # (2,1024) → (1,2,1024)
        elif arr.shape[1] == 2:     return arr.T[np.newaxis]     # (1024,2) → (1,2,1024)
    elif arr.ndim == 3:
        if arr.shape[1] == 2:       return arr                   # (N,2,1024) ✓
        elif arr.shape[0] == 2:     return arr.transpose(2,0,1)  # (2,1024,N)
    raise ValueError(f'无法解析形状: {arr.shape}')


# ── 功率归一化函数 ─────────────────────────────────────────────────
def power_normalize(signal_data: np.ndarray) -> np.ndarray:
    """功率归一化"""
    I, Q = signal_data[0], signal_data[1]
    power = np.mean(I ** 2 + Q ** 2)
    if power < 1e-10:
        return signal_data.astype(np.float32)
    return (signal_data / np.sqrt(power)).astype(np.float32)


def power_normalize_dataset(X: np.ndarray) -> np.ndarray:
    """对整个数据集 (N, 2, 1024) 逐帧进行功率归一化"""
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        X_norm[i] = power_normalize(X[i])
    return X_norm


# ╔══════════════════╗
# ║  2. 特征计算      ║
# ╚══════════════════╝

def seg_to_iq(X: np.ndarray):
    I = X[:, 0, :].ravel()
    Q = X[:, 1, :].ravel()
    return I, Q


def compute_stats(X: np.ndarray) -> dict:
    I, Q   = seg_to_iq(X)
    amp    = np.sqrt(I**2 + Q**2)
    pwr    = I**2 + Q**2
    return {
        'mean_pwr':  float(np.mean(pwr)),
        'std_pwr':   float(np.std(pwr)),
        'mean_amp':  float(np.mean(amp)),
        'std_amp':   float(np.std(amp)),
        'kurtosis':  float(stats.kurtosis(amp)),
        'skewness':  float(stats.skew(amp)),
        'dc_I':      float(np.mean(I)),
        'dc_Q':      float(np.mean(Q)),
        'papr_db':   float(10*np.log10(np.max(pwr)/(np.mean(pwr)+1e-12))),
        'n':         len(X),
    }


def compute_psd(X: np.ndarray, nfft=512, max_n=200) -> tuple:
    n   = min(len(X), max_n)
    psd = np.zeros(nfft)
    for i in range(n):
        sig = X[i,0,:nfft] + 1j*X[i,1,:nfft]
        win = np.hanning(nfft)
        P   = np.abs(np.fft.fft(sig*win, n=nfft))**2 / nfft
        psd += P
    psd = np.fft.fftshift(10*np.log10(psd/n + 1e-12))
    freqs = np.linspace(-0.5, 0.5, nfft)
    return freqs, psd


def compute_amp_hist(X: np.ndarray, bins=60):
    I, Q = seg_to_iq(X)
    amp  = np.sqrt(I**2 + Q**2)
    cnt, edges = np.histogram(amp, bins=bins, density=True)
    return (edges[:-1]+edges[1:])/2, cnt


def compute_phase_hist(X: np.ndarray, bins=72):
    I, Q  = seg_to_iq(X)
    phase = np.arctan2(Q, I)
    cnt, edges = np.histogram(phase, bins=bins, range=(-np.pi,np.pi), density=True)
    return (edges[:-1]+edges[1:])/2*180/np.pi, cnt


# ╔══════════════════╗
# ║  3. 诊断         ║
# ╚══════════════════╝

def diagnose(ss: dict, rs: dict) -> list:
    issues = []
    def chk(name, sv, rv, t_ok, t_warn, fmt='.4f'):
        d   = abs(sv - rv)
        rel = d / (abs(sv)+1e-9) * 100
        lvl = 'ok' if d < t_ok else ('warn' if d < t_warn else 'bad')
        issues.append((lvl, name,
            f"仿真={sv:{fmt}}  实采={rv:{fmt}}  差异={d:{fmt}} ({rel:.1f}%)"))
    chk('均值功率',   ss['mean_pwr'],  rs['mean_pwr'],  0.05, 0.25)
    chk('幅度标准差', ss['std_amp'],   rs['std_amp'],   0.03, 0.12)
    chk('峰度',       ss['kurtosis'],  rs['kurtosis'],  0.5,  2.0)
    chk('偏度',       ss['skewness'],  rs['skewness'],  0.2,  0.8)
    chk('PAPR(dB)',   ss['papr_db'],   rs['papr_db'],   1.0,  3.0, '.2f')
    dc = np.sqrt(rs['dc_I']**2 + rs['dc_Q']**2)
    lvl = 'bad' if dc>0.05 else ('warn' if dc>0.01 else 'ok')
    issues.append((lvl, 'DC偏移',
        f"实采DC幅度={dc:.4f}  I={rs['dc_I']:.4f}  Q={rs['dc_Q']:.4f}"))
    return issues


# ╔══════════════════╗
# ║  4. 绘图         ║
# ╚══════════════════╝

ICON  = {'ok':'✓', 'warn':'△', 'bad':'✗'}
LABEL = {'ok':'匹配', 'warn':'偏差', 'bad':'显著差异'}
COLOR = {'ok': C_OK, 'warn': C_WARN, 'bad': C_BAD}


def plot_compare(X_sim, X_real, X_sim_norm, X_real_norm, ss_raw, rs_raw, ss_norm, rs_norm, issues_raw, mod, snr, outdir):
    """绘制整合对比图（包含原始与功率归一化后的数据对比）"""
    fig = plt.figure(figsize=(18, 17))
    fig.suptitle(f'仿真 vs 实采对比分析  |  {mod}  SNR={snr:+.0f}dB',
                 fontsize=14, fontweight='bold', y=0.99)
    
    # 调整网格布局为 4 行 4 列
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.44, wspace=0.36)

    # ── 行0: 原始星座图对比 ──────────────────────────
    for col, (X, lbl, col_) in enumerate([
            (X_sim,  f'原始仿真星座图  ({len(X_sim)} 帧)',  C_SIM),
            (X_real, f'原始实采星座图  ({len(X_real)} 帧)', C_REAL)]):
        ax = fig.add_subplot(gs[0, col*2:(col+1)*2])
        I = X[:min(8,len(X)), 0, :].ravel()
        Q = X[:min(8,len(X)), 1, :].ravel()
        ax.scatter(I, Q, s=1.5, alpha=0.3, color=col_, rasterized=True)
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.set_xlabel('I', fontsize=8); ax.set_ylabel('Q', fontsize=8)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.2, lw=0.5); ax.tick_params(labelsize=7)

    # ── 行1: 归一化星座图对比 ────────────────────────
    for col, (X, lbl, col_) in enumerate([
            (X_sim_norm,  f'归一化仿真星座图  ({len(X_sim_norm)} 帧)',  C_SIM),
            (X_real_norm, f'归一化实采星座图  ({len(X_real_norm)} 帧)', C_REAL)]):
        ax = fig.add_subplot(gs[1, col*2:(col+1)*2])
        I = X[:min(8,len(X)), 0, :].ravel()
        Q = X[:min(8,len(X)), 1, :].ravel()
        ax.scatter(I, Q, s=1.5, alpha=0.3, color=col_, rasterized=True)
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.set_xlabel('I', fontsize=8); ax.set_ylabel('Q', fontsize=8)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.2, lw=0.5); ax.tick_params(labelsize=7)

    # ── 行2: 归一化后 PSD 与 幅度分布对比 ─────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    for X, lbl, col_ in [(X_sim_norm,'仿真 (归一化)',C_SIM),(X_real_norm,'实采 (归一化)',C_REAL)]:
        f, p = compute_psd(X)
        ax2.plot(f, p, color=col_, lw=1.3, label=lbl)
        ax2.fill_between(f, p, p.min()-3, alpha=0.1, color=col_)
    ax2.set_title('功率谱密度对比 (PSD, 归一化)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('归一化频率', fontsize=8); ax2.set_ylabel('dB', fontsize=8)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2, lw=0.5); ax2.tick_params(labelsize=7)

    ax3 = fig.add_subplot(gs[2, 2:])
    # 绘制原始分布（虚线）
    cx_s_raw, cy_s_raw = compute_amp_hist(X_sim)
    cx_r_raw, cy_r_raw = compute_amp_hist(X_real)
    ax3.plot(cx_s_raw, cy_s_raw, color=C_SIM, lw=1.1, linestyle='--', label='仿真 (原始)')
    ax3.plot(cx_r_raw, cy_r_raw, color=C_REAL, lw=1.1, linestyle='--', label='实采 (原始)')
    # 绘制归一化分布（实线+填充）
    cx_s_norm, cy_s_norm = compute_amp_hist(X_sim_norm)
    cx_r_norm, cy_r_norm = compute_amp_hist(X_real_norm)
    ax3.plot(cx_s_norm, cy_s_norm, color=C_SIM, lw=1.5, label='仿真 (归一化)')
    ax3.plot(cx_r_norm, cy_r_norm, color=C_REAL, lw=1.5, label='实采 (归一化)')
    ax3.fill_between(cx_s_norm, cy_s_norm, alpha=0.08, color=C_SIM)
    ax3.fill_between(cx_r_norm, cy_r_norm, alpha=0.08, color=C_REAL)
    
    ax3.set_title('幅度概率分布 (原始 vs 归一化)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('幅度', fontsize=8); ax3.set_ylabel('概率密度', fontsize=8)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.2, lw=0.5); ax3.tick_params(labelsize=7)

    # ── 行3: 相位分布 / 归一化统计柱图 / 原始数据诊断 ───────
    ax4 = fig.add_subplot(gs[3, 0])
    for X, lbl, col_ in [(X_sim_norm,'仿真',C_SIM),(X_real_norm,'实采',C_REAL)]:
        cx, cy = compute_phase_hist(X)
        ax4.plot(cx, cy, color=col_, lw=1.3, label=lbl)
        ax4.fill_between(cx, cy, alpha=0.12, color=col_)
    ax4.set_title('相位分布', fontsize=10, fontweight='bold')
    ax4.set_xlabel('相位 (°)', fontsize=8); ax4.set_ylabel('概率密度', fontsize=8)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.2, lw=0.5); ax4.tick_params(labelsize=7)

    # 归一化后的统计数据柱图比较
    ax5 = fig.add_subplot(gs[3, 1:3])
    keys   = ['mean_pwr','std_amp','kurtosis','skewness','papr_db']
    labels = ['均值功率','幅度σ','峰度','偏度','PAPR(dB)']
    sv = np.array([ss_norm[k] for k in keys])
    rv = np.array([rs_norm[k] for k in keys])
    mx = np.maximum(np.abs(sv), np.abs(rv)) + 1e-9
    x  = np.arange(len(labels)); w = 0.35
    b1 = ax5.bar(x-w/2, sv/mx, w, color=C_SIM,  label='仿真', alpha=0.85)
    b2 = ax5.bar(x+w/2, rv/mx, w, color=C_REAL, label='实采', alpha=0.85)
    for bar, v in zip(b1, sv):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=6, color=C_SIM)
    for bar, v in zip(b2, rv):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=6, color=C_REAL)
    ax5.set_xticks(x); ax5.set_xticklabels(labels, fontsize=8)
    ax5.set_title('归一化后特征对比（标注归一化后原值）', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=8); ax5.grid(axis='y', alpha=0.2, lw=0.5)
    ax5.tick_params(labelsize=7)

    # 诊断文字显示（依据原始未归一化的特征，以暴露量化/缩放问题）
    ax6 = fig.add_subplot(gs[3, 3])
    ax6.axis('off')
    lines = [f'【{mod} 原始数据诊断】\n']
    for lvl, field, msg in issues_raw:
        lines.append(f"{ICON[lvl]} {field}: {LABEL[lvl]}")
    ax6.text(0.02, 0.98, '\n'.join(lines), transform=ax6.transAxes,
             fontsize=8, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='#f5f5f5', alpha=0.9))
    for i, (lvl, field, msg) in enumerate(issues_raw):
        ax6.text(0.02, 0.72-i*0.09, f'  {msg}',
                 transform=ax6.transAxes, fontsize=6.2,
                 color=COLOR[lvl], va='top')

    fname = os.path.join(outdir, f'compare_{mod}_snr{snr:.0f}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 对比图 (包含功率归一化): {fname}')
    return fname


def plot_scatter_only(X_sim, X_real, X_sim_norm, X_real_norm, mod, snr, outdir):
    """单独输出大幅高清星座图，并排展示原始与功率归一化后的状态"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(f'星座图分析（原始 vs 功率归一化）  |  {mod}  SNR={snr:+.0f}dB',
                 fontsize=13, fontweight='bold')
    
    # 第一行：原始数据
    for col, (X, lbl, col_) in enumerate([
            (X_sim,  f'原始仿真  ({len(X_sim)} 帧)',  C_SIM),
            (X_real, f'原始实采  ({len(X_real)} 帧)', C_REAL)]):
        ax = axes[0, col]
        I = X[:, 0, :].ravel(); Q = X[:, 1, :].ravel()
        ax.scatter(I, Q, s=0.8, alpha=0.25, color=col_, rasterized=True)
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel('I', fontsize=9); ax.set_ylabel('Q', fontsize=9)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.2, lw=0.5)

    # 第二行：归一化数据
    for col, (X, lbl, col_) in enumerate([
            (X_sim_norm,  f'归一化仿真  ({len(X_sim_norm)} 帧)',  C_SIM),
            (X_real_norm, f'归一化实采  ({len(X_real_norm)} 帧)', C_REAL)]):
        ax = axes[1, col]
        I = X[:, 0, :].ravel(); Q = X[:, 1, :].ravel()
        ax.scatter(I, Q, s=0.8, alpha=0.25, color=col_, rasterized=True)
        ax.set_title(lbl, fontsize=11, fontweight='bold')
        ax.set_xlabel('I', fontsize=9); ax.set_ylabel('Q', fontsize=9)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.2, lw=0.5)

    plt.tight_layout()
    fname = os.path.join(outdir, f'constellation_{mod}_snr{snr:.0f}.png')
    fig.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  → 高清星座图 (原始+归一化): {fname}')


def print_report(ss_raw, rs_raw, ss_norm, rs_norm, issues_raw, issues_norm, mod, snr):
    sep = '─' * 62
    print(f'\n{sep}')
    print(f'  诊断报告 (原始数据)  |  {mod}  SNR={snr:+.0f}dB')
    print(f'  仿真样本: {ss_raw["n"]:,}    实采样本: {rs_raw["n"]:,}')
    print(sep)
    for lvl, field, msg in issues_raw:
        print(f'  {ICON[lvl]} [{LABEL[lvl]:4s}]  {field}: {msg}')
    print(sep)
    print(f'  诊断报告 (功率归一化后)')
    print(sep)
    for lvl, field, msg in issues_norm:
        print(f'  {ICON[lvl]} [{LABEL[lvl]:4s}]  {field}: {msg}')
    print(sep)
    
    bad  = [i for i in issues_raw if i[0]=='bad']
    warn = [i for i in issues_raw if i[0]=='warn']
    print(f'  汇总: {len(bad)} 项显著差异  {len(warn)} 项轻微偏差')
    
    if bad:
        print('\n  ► 重点排查建议:')
        for _, field, _ in bad:
            if '功率' in field or '幅度' in field:
                print('    • 功率/幅度差异：请查阅归一化后的数据。若归一化后诊断表现良好，')
                print('      说明两端系统存在固定尺度缩放或模数转换 (ADC) 的增益不一致现象。')
            if '峰度' in field:
                print('    • 峰度差异：128QAM cross-QAM 星座点分布与实际设备可能不同')
                print('      建议对比设备文档中的星座映射方式')
            if 'DC' in field:
                print('    • DC偏移：建议实采预处理时减去均值（去直流）')
            if 'PAPR' in field:
                print('    • PAPR差异：仿真PA非线性参数 (IBO/p) 可能需要调整')
    print(sep)


# ╔══════════════════╗
# ║  5. 主流程        ║
# ╚══════════════════╝

def main():
    parser = argparse.ArgumentParser(description='仿真 vs 实采 IQ 对比分析')
    parser.add_argument('--sim',      required=True, help='gen_single_mod.py 生成的 .npz')
    parser.add_argument('--real',     required=True, help='实采数据目录或单文件')
    parser.add_argument('--outdir',   default='./compare_results', help='输出目录')
    parser.add_argument('--max-real', type=int, default=500, help='最大实采片段数')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1. 加载数据
    X_sim,  mod, snr = load_sim(args.sim)
    X_real           = load_real(args.real, max_n=args.max_real)

    # 2. 功率归一化处理
    X_sim_norm  = power_normalize_dataset(X_sim)
    X_real_norm = power_normalize_dataset(X_real)

    # 3. 计算统计指标（原始 vs 归一化后）
    ss_raw  = compute_stats(X_sim)
    rs_raw  = compute_stats(X_real)
    ss_norm = compute_stats(X_sim_norm)
    rs_norm = compute_stats(X_real_norm)

    # 4. 诊断评估
    issues_raw  = diagnose(ss_raw, rs_raw)
    issues_norm = diagnose(ss_norm, rs_norm)

    # 5. 输出报告与画图
    print_report(ss_raw, rs_raw, ss_norm, rs_norm, issues_raw, issues_norm, mod, snr)
    plot_compare(X_sim, X_real, X_sim_norm, X_real_norm, ss_raw, rs_raw, ss_norm, rs_norm, issues_raw, mod, snr, args.outdir)
    plot_scatter_only(X_sim, X_real, X_sim_norm, X_real_norm, mod, snr, args.outdir)

    print(f'\n✓ 完成，结果保存至: {args.outdir}/')


if __name__ == '__main__':
    main()