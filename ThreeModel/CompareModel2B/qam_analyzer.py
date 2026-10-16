"""
qam_analyzer.py — QAM 信号验证工具
====================================
验证 VSG25A 发送的 QAM 信号是否正确。
输入：RSPduo IQ-OUT WAV（SPS=4，即 FS_RX = SYM_RATE × 4）

分析内容：
  1. IQ 时域波形
  2. Welch 功率谱（验证 RRC 成形和带宽）
  3. IQ 分量对称性（DC偏置、幅度比、正交性）
  4. 校正前/后星座图（支持 16/64/256 QAM）
  5. EVM 计算（误差向量幅度）
  6. MER 计算（调制误差率）
  7. 眼图（可选，仅 BPSK/QPSK 有意义）

用法：
  python qam_analyzer.py --wav capture_64qam.wav --mod 64QAM --fs 400000 --sr 100000
  python qam_analyzer.py --wav capture_256qam.wav --mod 256QAM
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

# ================= 终极字体修复方案 =================
# 1. 设置主字体（优先使用支持中英文混合+特殊符号的 微软雅黑/苹方）
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',     # Windows 微软雅黑 (英文和符号显示非常漂亮)
    'PingFang SC',         # Mac 苹方
    'Arial Unicode MS',    # 备用的全能 Unicode 字体
    'SimHei',              # Windows 黑体 fallback
    'DejaVu Sans',         # Matplotlib 自带的英文默认字体
    'sans-serif'
]

# 2. 修复代码中强制使用 monospace（等宽）导致的中文丢失问题
plt.rcParams['font.monospace'] = [
    'Consolas',            # 优先使用好看的英文等宽
    'Microsoft YaHei',     # 当等宽字体遇到中文/符号时，退回使用微软雅黑
    'SimHei',
    'monospace'
]

# 3. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# =================================================

try:
    import soundfile as sf
except ImportError:
    raise ImportError("pip install soundfile")


# ─────────────────────────────────────────────────────────────────────────────
# 默认参数（命令行可覆盖）
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WAV     = 'E:/SUN/Right/64QAM_05_SDRuno_20260421_112901Z_99900kHz.wav'
DEFAULT_MOD     = '64QAM'      # 16QAM / 64QAM / 256QAM
DEFAULT_FS_RX   = 2000_000      # RSPduo 采样率（必须 = SYM_RATE × 4）
DEFAULT_SR      = 500_000      # 符号速率
DEFAULT_ALPHA   = 0.35         # RRC 滚降系数
DEFAULT_SPAN    = 8            # RRC 滤波器跨度（符号数）
SPS_TARGET      = 4            # 训练 SPS，也是期望的接收 SPS


# ─────────────────────────────────────────────────────────────────────────────
# 星座点生成
# ─────────────────────────────────────────────────────────────────────────────
def qam_constellation(M: int) -> np.ndarray:
    """生成方形 QAM 星座点（格雷码排列，单位平均功率）"""
    k = int(np.sqrt(M))
    if k * k != M:
        raise ValueError(f"M={M} 必须是完全平方数（16/64/256）")
    levels = np.arange(-(k-1), k, 2)          # [-3,-1,+1,+3] for 16QAM
    I, Q   = np.meshgrid(levels, levels)
    syms   = (I + 1j * Q).flatten()
    syms  /= np.sqrt(np.mean(np.abs(syms)**2))  # 归一化到单位平均功率
    return syms

PQPSK = np.array([(1+1j),(1-1j),(-1+1j),(-1-1j)]) / np.sqrt(2)

MOD_TABLE = {
    'BPSK':   np.array([1+0j, -1+0j]),
    'QPSK':   PQPSK,
    '16QAM':  qam_constellation(16),
    '64QAM':  qam_constellation(64),
    '256QAM': qam_constellation(256),
}


# ─────────────────────────────────────────────────────────────────────────────
# RRC 滤波器
# ─────────────────────────────────────────────────────────────────────────────
def rrc_filter(alpha: float, sps: int, span: int) -> np.ndarray:
    n = span * sps + 1
    t = np.arange(-(span*sps)//2, (span*sps)//2+1) / sps
    h = np.zeros(n)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 + alpha*(4/np.pi - 1)
        elif abs(abs(ti) - 1/(4*alpha)) < 1e-9:
            h[i] = (alpha/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*alpha))+
                                        (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            den = np.pi*ti*(1-(4*alpha*ti)**2)
            h[i] = num/den
    h /= np.sqrt(np.sum(h**2))
    return h


# ─────────────────────────────────────────────────────────────────────────────
# 频偏估计（M次方法，适用于PSK；QAM用导频版）
# ─────────────────────────────────────────────────────────────────────────────
def freq_correct_power(iq: np.ndarray, fs: int, M: int = 4):
    """M次方法：适用于PSK。QAM时M=4仅能去除90°模糊，频偏估计效果有限"""
    spec = np.abs(np.fft.fft(iq**M, len(iq)))
    freq = np.fft.fftfreq(len(iq), 1/fs)
    foff = freq[np.argmax(spec)] / M
    t    = np.arange(len(iq)) / fs
    return iq * np.exp(-1j * 2*np.pi * foff * t), foff


# ─────────────────────────────────────────────────────────────────────────────
# QAM 符号定时恢复 + 相位模糊消除（4重对称）
# ─────────────────────────────────────────────────────────────────────────────
def demodulate_qam(iq_fc: np.ndarray, sps: int, ideal: np.ndarray,
                   rrc_h: np.ndarray):
    """
    接收匹配滤波 + 穷举定时偏移 + 穷举4种相位旋转
    返回：最优符号序列，MF后信号，振幅缩放因子，定时偏移，相位旋转
    """
    iq_mf = np.convolve(iq_fc, rrc_h, mode='same')

    # 幅度归一化：使符号功率与理想星座一致
    avg_rx_pwr = np.mean(np.abs(iq_mf)**2)
    avg_id_pwr = np.mean(np.abs(ideal)**2)
    scale = np.sqrt(avg_id_pwr / (avg_rx_pwr + 1e-10))
    iq_mf_n = iq_mf * scale

    best_evm = 1e9
    best_syms = None
    best_off  = 0
    best_rot  = 0

    for off in range(sps):
        s = iq_mf_n[off::sps]
        # 振幅自适应归一化（每次）
        s_n = s / np.sqrt(np.mean(np.abs(s)**2) / avg_id_pwr + 1e-10)
        for rot in range(4):
            s2   = s_n * np.exp(1j * rot * np.pi/2)
            dsts = np.array([np.abs(s2 - p)**2 for p in ideal])
            ref  = ideal[np.argmin(dsts, axis=0)]
            evm  = np.sqrt(np.mean(np.abs(s2-ref)**2) / (np.mean(np.abs(ref)**2)+1e-10))
            if evm < best_evm:
                best_evm  = evm
                best_off  = off
                best_rot  = rot
                best_syms = s2

    return best_syms, iq_mf, scale, best_off, best_rot


# ─────────────────────────────────────────────────────────────────────────────
# 质量指标
# ─────────────────────────────────────────────────────────────────────────────
def quality_metrics(syms: np.ndarray, ideal: np.ndarray):
    dsts = np.array([np.abs(syms - p)**2 for p in ideal])
    ref  = ideal[np.argmin(dsts, axis=0)]
    err  = syms - ref
    evm  = np.sqrt(np.mean(np.abs(err)**2) / np.mean(np.abs(ref)**2)) * 100
    mer  = 10 * np.log10(np.mean(np.abs(ref)**2) / (np.mean(np.abs(err)**2)+1e-20))
    return evm, mer, ref


# ─────────────────────────────────────────────────────────────────────────────
# IQ 分量健康检查
# ─────────────────────────────────────────────────────────────────────────────
def iq_health_check(iq: np.ndarray):
    I, Q = iq.real, iq.imag
    dc_i = np.mean(I)
    dc_q = np.mean(Q)
    std_i = np.std(I)
    std_q = np.std(Q)
    amp_ratio = std_i / (std_q + 1e-10)
    iq_corr   = np.mean(I * Q) / (std_i * std_q + 1e-10)
    pwr_dbfs  = 10 * np.log10(np.mean(I**2 + Q**2) + 1e-20)

    print(f"\n  [IQ Health]")
    print(f"  功率         : {pwr_dbfs:.1f} dBFS")
    print(f"  DC offset I  : {dc_i:.5f}  {'✓' if abs(dc_i) < 0.02 else '⚠ 偏大'}")
    print(f"  DC offset Q  : {dc_q:.5f}  {'✓' if abs(dc_q) < 0.02 else '⚠ 偏大'}")
    print(f"  幅度比 I/Q   : {amp_ratio:.4f}  {'✓' if 0.9<amp_ratio<1.1 else '⚠ IQ 不平衡'}")
    print(f"  IQ 相关性    : {iq_corr:.4f}  {'✓' if abs(iq_corr) < 0.05 else '⚠ 非正交'}")
    return dc_i, dc_q, amp_ratio, iq_corr


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────
def plot_analysis(iq_raw, iq_fc, iq_mf, syms, ref_syms,
                  fs, sps, foff, evm, mer, mod, alpha, outpath):
    BG='#0d1117'; AB='#161b22'; GD='#21262d'
    W='#e6edf3';  DM='#8b949e'
    CI='#58a6ff'; CQ='#ffa657'; CS='#3fb950'
    CC='#ff7b72'; CE='#bc8cff'; CO='#f78166'

    def sx(ax, title=''):
        ax.set_facecolor(AB)
        for sp in ax.spines.values(): sp.set_color(GD)
        ax.tick_params(colors=DM, labelsize=7)
        ax.grid(True, color=GD, lw=0.4, alpha=0.7)
        if title: ax.set_title(title, color=W, fontsize=8, fontweight='bold', pad=4)

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.44, wspace=0.35,
                            left=0.055, right=0.97, top=0.92, bottom=0.055)

    # ── 面板1：IQ 时域 ────────────────────────────────────────────────────────
    Nt   = min(int(fs * 0.002), len(iq_raw))
    tms  = np.arange(Nt) / fs * 1000
    ax1  = fig.add_subplot(gs[0, :2])
    ax1.plot(tms, iq_raw[:Nt].real, color=CI, lw=0.8, label='I')
    ax1.plot(tms, iq_raw[:Nt].imag, color=CQ, lw=0.8, label='Q', alpha=0.85)
    ax1.set_xlabel('Time (ms)', color=DM, fontsize=8)
    ax1.set_ylabel('Amplitude', color=DM, fontsize=8)
    ax1.legend(fontsize=7, facecolor=AB, edgecolor=GD, labelcolor=W)
    sx(ax1, 'IQ 时域波形（前 2 ms）')

    # ── 面板2：Welch PSD ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:])
    fw, Pxx = welch(iq_raw, fs=fs, nperseg=2048, return_onesided=False)
    idx2 = np.argsort(fw)
    fw, Pd = fw[idx2], 10*np.log10(Pxx[idx2]+1e-20)
    ax2.plot(fw/1e3, Pd, color=CS, lw=0.9)
    ax2.fill_between(fw/1e3, Pd.min()-5, Pd, color=CS, alpha=0.1)
    bw = (DEFAULT_SR*(1+alpha)/2) if hasattr(DEFAULT_SR,'__float__') else 67500
    ax2.axvspan(-bw/1e3, bw/1e3, color=CI, alpha=0.07)
    ax2.set_xlabel('Frequency (kHz)', color=DM, fontsize=8)
    ax2.set_ylabel('PSD (dBFS/Hz)', color=DM, fontsize=8)
    ax2.set_xlim(-fs/2/1e3, fs/2/1e3)
    sx(ax2, f'Welch PSD  |  Δf={foff:+.0f} Hz')

    # ── 面板3：校正前星座图 ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    rs   = iq_mf[::sps]
    Nc   = min(2000, len(rs))
    ax3.scatter(rs[:Nc].real, rs[:Nc].imag, color=CC, s=4, alpha=0.3, linewidths=0)
    ax3.axhline(0, color=GD, lw=0.6); ax3.axvline(0, color=GD, lw=0.6)
    lim = max(np.percentile(np.abs(rs[:Nc]), 98)*1.5, 1.2)
    ax3.set_xlim(-lim,lim); ax3.set_ylim(-lim,lim); ax3.set_aspect('equal')
    ax3.set_xlabel('I', color=DM, fontsize=8); ax3.set_ylabel('Q', color=DM, fontsize=8)
    sx(ax3, '星座图（频偏/相位校正前）')

    # ── 面板4：校正后星座图 ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    Nc2  = min(3000, len(syms))
    ideal = MOD_TABLE[mod]
    sc   = ax4.scatter(syms[:Nc2].real, syms[:Nc2].imag,
                       c=np.angle(syms[:Nc2]), cmap='hsv', s=6, alpha=0.45, linewidths=0)
    ax4.scatter(ideal.real, ideal.imag, color='white', s=60, marker='+', zorder=5, lw=1.5)
    ax4.axhline(0, color=GD, lw=0.6); ax4.axvline(0, color=GD, lw=0.6)
    lim2 = np.max(np.abs(ideal))*1.4
    ax4.set_xlim(-lim2,lim2); ax4.set_ylim(-lim2,lim2); ax4.set_aspect('equal')
    ax4.set_xlabel('I', color=DM, fontsize=8); ax4.set_ylabel('Q', color=DM, fontsize=8)
    cb = plt.colorbar(sc, ax=ax4); cb.set_label('Phase', color=DM, fontsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=DM)
    sx(ax4, f'{mod} 星座图（校正后）   EVM={evm:.2f}%  MER={mer:.1f}dB')

    # ── 面板5：误差向量分布 ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    err  = syms[:Nc2] - ref_syms[:Nc2]
    ax5.scatter(err.real, err.imag, color=CE, s=5, alpha=0.4, linewidths=0)
    ax5.axhline(0, color=GD, lw=0.8); ax5.axvline(0, color=GD, lw=0.8)
    r_max = np.percentile(np.abs(err), 99)*1.5
    ax5.set_xlim(-r_max,r_max); ax5.set_ylim(-r_max,r_max); ax5.set_aspect('equal')
    ax5.set_xlabel('误差 I', color=DM, fontsize=8); ax5.set_ylabel('误差 Q', color=DM, fontsize=8)
    sx(ax5, f'误差向量分布  （点越集中越好）')

    # ── 面板6：汇总报告 ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    sx(ax6, '分析报告')
    ok   = evm < {'16QAM':4,'64QAM':2.5,'256QAM':1.5}.get(mod, 8)
    snr  = mer + {'16QAM':4,'64QAM':5,'256QAM':6}.get(mod, 3)
    I, Q = iq_raw.real, iq_raw.imag
    rows = [
        ('调制方式',    mod),
        ('载波频率',    '100.000 MHz'),
        ('符号速率',    f'{DEFAULT_SR/1e3:.0f} kSym/s'),
        ('RRC α',      f'{alpha}'),
        ('RX 采样率',  f'{fs/1e6:.3f} MHz'),
        ('SPS',        f'{sps}'),
        ('频偏（估）', f'{foff:+.1f} Hz'),
        ('EVM（rms）', f'{evm:.2f} %'),
        ('MER',        f'{mer:.1f} dB'),
        ('估算 SNR',   f'~{snr:.1f} dB'),
        ('DC offset I',f'{np.mean(I):.4f}'),
        ('DC offset Q',f'{np.mean(Q):.4f}'),
        ('IQ 幅度比',  f'{np.std(I)/(np.std(Q)+1e-10):.4f}'),
        ('',           ''),
        ('信号状态',   'OK — 信号正常' if ok else '⚠ EVM 偏高，检查硬件'),
    ]
    for i, (k, v) in enumerate(rows):
        vc = '#3fb950' if (k=='信号状态' and ok) else \
             '#f85149' if k=='信号状态' else W
        ax6.text(0.02, 0.97-i*0.062, (k+' :') if k else '',
                 fontsize=8, color=DM, transform=ax6.transAxes, va='top')
        ax6.text(0.50, 0.97-i*0.062, v, fontsize=8, fontweight='bold',
                 color=vc, transform=ax6.transAxes, va='top')

    fig.suptitle(f'{mod} 信号分析  |  VSG25A → RSPduo',
                 color=W, fontsize=13, fontweight='bold', y=0.965)
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\n  图表保存 → {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='QAM 信号分析工具')
    parser.add_argument('--wav',   type=str,   default=DEFAULT_WAV)
    parser.add_argument('--mod',   type=str,   default=DEFAULT_MOD,
                        choices=['BPSK','QPSK','16QAM','64QAM','256QAM'])
    parser.add_argument('--fs',    type=int,   default=DEFAULT_FS_RX)
    parser.add_argument('--sr',    type=int,   default=DEFAULT_SR)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--out',   type=str,   default='qam_analysis.png')
    args = parser.parse_args()

    fs, sr, alpha, mod = args.fs, args.sr, args.alpha, args.mod
    sps = fs // sr
    rrc_h = rrc_filter(alpha, sps, DEFAULT_SPAN)
    ideal = MOD_TABLE[mod]

    print(f"\n{'='*60}")
    print(f"  QAM Analyzer  |  {mod}  |  FS={fs}  SR={sr}  SPS={sps}")
    print(f"{'='*60}")

    if sps < 2:
        print(f"[Error] SPS={sps} 太小，采样率需 ≥ 2×符号速率")
        return

    # 读取 WAV
    if not os.path.isfile(args.wav):
        print(f"[Error] 文件不存在: {args.wav}"); return
    data, fs_actual = sf.read(args.wav, always_2d=True, dtype='float32')
    if data.shape[1] < 2:
        print("[Error] 需要立体声 WAV"); return
    iq_raw = data[:,0] + 1j*data[:,1]
    if int(fs_actual) != fs:
        print(f"  ⚠ WAV 实际 fs={fs_actual}，配置 fs={fs}")
    print(f"  WAV: {args.wav}  ({len(iq_raw)/fs_actual:.2f} s)")

    # 取前 0.2s 分析
    N_anal = min(int(fs * 0.2), len(iq_raw))
    iq_raw = iq_raw[:N_anal]

    # IQ 健康检查
    iq_health_check(iq_raw)

    # 去直流
    iq_dc = iq_raw - np.mean(iq_raw)

    # 频偏校正（PSK用4次方法；QAM此处仍用，精度有限）
    iq_fc, foff = freq_correct_power(iq_dc, fs, M=4)
    print(f"\n  [Freq] 估计频偏 = {foff:+.1f} Hz")

    # 解调
    syms, iq_mf, scale, t_off, rot = demodulate_qam(iq_fc, sps, ideal, rrc_h)
    print(f"  [Demod] 定时偏移={t_off}/{sps}  相位旋转={rot}×90°")

    # 质量指标
    evm, mer, ref_syms = quality_metrics(syms, ideal)
    snr = mer + {'16QAM':4,'64QAM':5,'256QAM':6}.get(mod,3)
    print(f"\n  [Quality]")
    print(f"  EVM    = {evm:.2f} %")
    print(f"  MER    = {mer:.1f} dB")
    print(f"  SNR 估 = ~{snr:.1f} dB")
    thresh = {'16QAM':4,'64QAM':2.5,'256QAM':1.5,'QPSK':8,'BPSK':12}.get(mod,8)
    print(f"  状态   : {'OK' if evm<thresh else '⚠ EVM 偏高'}")

    # 绘图
    plot_analysis(iq_raw, iq_fc, iq_mf, syms, ref_syms,
                  fs, sps, foff, evm, mer, mod, alpha, args.out)
    print("\n  Done.")

if __name__ == '__main__':
    main()
