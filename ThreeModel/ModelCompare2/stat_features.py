"""
stat_features.py v3
20维基础 + 22维新增 = 42维
核心改动：16-bin直方图 + Gini系数 + 功率域峰度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HigherOrderStats(nn.Module):

    RAW_DIM = 42  # ← 从32改为42

    def __init__(self, proj_dim=96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self.RAW_DIM, 192),   # ← 128→192，匹配更多输入
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
        )

    def forward(self, x):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            feats = self._extract(x.float())
        return self.proj(feats)

    def _extract(self, x):
        I, Q = x[..., 0], x[..., 1]
        B, L = I.shape
        dev = x.device

        # ══════════════════════════════════════
        #  原有 20 维特征（不变）
        # ══════════════════════════════════════
        amp = torch.sqrt(I ** 2 + Q ** 2 + 1e-10)
        amp_mu = amp.mean(1, keepdim=True)
        amp_sig = amp.std(1, keepdim=True) + 1e-10
        amp_n = (amp - amp_mu) / amp_sig

        f_amp_mu = amp_mu
        f_amp_sig = amp_sig
        f_skew = amp_n.pow(3).mean(1, keepdim=True)
        f_kurt = amp_n.pow(4).mean(1, keepdim=True)
        f_papr = amp.max(1, keepdim=True)[0] / (amp_mu + 1e-10)

        amp_sorted = amp.sort(1)[0]
        qi = [int(q * (L - 1)) for q in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)]
        f_quantiles = torch.stack([amp_sorted[:, i] for i in qi], dim=1)

        a2 = amp.pow(2).mean(1, keepdim=True)
        a4 = amp.pow(4).mean(1, keepdim=True)
        f_sigma_a = a4 / (a2.pow(2) + 1e-10)

        phase = torch.atan2(Q, I)
        dp = phase[:, 1:] - phase[:, :-1]
        dp = torch.atan2(torch.sin(dp), torch.cos(dp))
        f_phase_std = dp.std(1, keepdim=True)
        f_phase_mean = dp.abs().mean(1, keepdim=True)

        Ic = I - I.mean(1, keepdim=True)
        Qc = Q - Q.mean(1, keepdim=True)
        I_std = Ic.std(1, keepdim=True) + 1e-10
        Q_std = Qc.std(1, keepdim=True) + 1e-10
        f_iq_corr = (Ic * Qc).mean(1, keepdim=True) / (I_std * Q_std)

        f_above = (amp > amp_mu).float().mean(1, keepdim=True)
        amp_diff = amp_sorted[:, 1:] - amp_sorted[:, :-1]
        f_entropy_raw = amp_diff.std(1, keepdim=True)

        a6 = amp.pow(6).mean(1, keepdim=True)
        f_sigma_6 = a6 / (a2.pow(3) + 1e-10)

        f_I_kurt = ((Ic / I_std).pow(4)).mean(1, keepdim=True)
        f_Q_kurt = ((Qc / Q_std).pow(4)).mean(1, keepdim=True)

        # ══════════════════════════════════════
        #  新增 22 维特征
        # ══════════════════════════════════════

        # ── 16-bin 幅度直方图 (16维) ──
        # 64QAM 9种幅度 → 16bin中约9个有值，7个空
        # 256QAM 32种幅度 → 16bin中几乎全满
        N_BINS = 16
        amp_max_val = amp.max(1, keepdim=True)[0] + 1e-10
        amp_01 = (amp / amp_max_val).clamp(0, 1.0 - 1e-6)
        bin_idx = (amp_01 * N_BINS).long()
        f_hist = torch.zeros(B, N_BINS, device=dev, dtype=x.dtype)
        for i in range(N_BINS):
            f_hist[:, i] = (bin_idx == i).float().mean(1)
        # (B, 16)                                               # 21-36

        # ── 直方图熵 (1维) ──
        # 64QAM ≈ 1.5 (集中), 256QAM ≈ 2.5 (分散)
        f_hist_entropy = -(f_hist * torch.log(f_hist + 1e-10)).sum(
            1, keepdim=True)                                    # 37

        # ── Gini 系数 (1维) ──
        # 衡量分布的不均匀性：0=完全均匀, 1=完全集中
        # 64QAM: Gini 高 (少数bin占大部分), 256QAM: Gini 低
        hist_sorted = f_hist.sort(1)[0]
        n = N_BINS
        idx_weight = torch.arange(1, n + 1, device=dev, dtype=x.dtype)
        f_gini = (2 * (idx_weight * hist_sorted).sum(1, keepdim=True) /
                  (n * hist_sorted.sum(1, keepdim=True) + 1e-10) -
                  (n + 1) / n)                                  # 38

        # ── 直方图峰计数 (1维) ──
        # 有多少个bin的值明显高于其邻居
        hist_pad = F.pad(f_hist, (1, 1), value=0)
        is_peak = ((hist_pad[:, 1:-1] > hist_pad[:, :-2]) &
                   (hist_pad[:, 1:-1] > hist_pad[:, 2:]))
        f_n_peaks = is_peak.float().sum(1, keepdim=True)        # 39

        # ── 幅度间隔统计 (2维) ──
        step = max(1, L // 256)
        amp_sub = amp_sorted[:, ::step]
        gaps = amp_sub[:, 1:] - amp_sub[:, :-1]
        f_gap_mean = gaps.mean(1, keepdim=True)                 # 40
        f_gap_std = gaps.std(1, keepdim=True)                   # 41

        # ── 聚类数估计 (1维) ──
        gap_threshold = f_gap_mean + 2.0 * f_gap_std
        f_n_clusters = (gaps > gap_threshold).float().sum(
            1, keepdim=True)                                    # 42

        # ══════════════════════════════════════
        #  拼接所有 42 维
        # ══════════════════════════════════════
        return torch.cat([
            # 原 20 维
            f_amp_mu, f_amp_sig, f_skew, f_kurt, f_papr,
            f_quantiles,
            f_sigma_a,
            f_phase_std, f_phase_mean,
            f_iq_corr,
            f_above, f_entropy_raw,
            f_sigma_6,
            f_I_kurt, f_Q_kurt,
            # 新 22 维
            f_hist,
            f_hist_entropy,
            f_gini,
            f_n_peaks,
            f_gap_mean, f_gap_std,
            f_n_clusters,
        ], dim=1)  # (B, 42)