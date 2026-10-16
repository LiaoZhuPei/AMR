"""
stat_features.py
高阶统计特征提取 —— 专门针对 QAM 阶数区分
思路: 显式计算幅度分布统计量，让网络不必从零学习"峰度"这种高阶概念
"""
import torch
import torch.nn as nn


class HigherOrderStats(nn.Module):
    """
    从原始 IQ 提取 20 维统计特征 → 映射到 proj_dim 维

    QAM 阶数的本质差异:
      - 幅度峰度随 QAM 阶数增大而变化
      - 四阶矩比 σ_a = E[a⁴]/E[a²]² 对不同 QAM 有不同理论值
      - 幅度分位数曲线形状不同 (分位数越多, 区分越精细)
    """

    RAW_DIM = 20

    def __init__(self, proj_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self.RAW_DIM, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """x: (B, L, 2)  功率归一化后的 IQ"""
        # 统计量本身不需要梯度, 但投影层需要
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            feats = self._extract(x.float())
        return self.proj(feats)

    def _extract(self, x):
        I, Q = x[..., 0], x[..., 1]          # (B, L)
        B, L = I.shape

        # ══════ 幅度统计 (QAM 区分核心) ══════
        amp = torch.sqrt(I ** 2 + Q ** 2 + 1e-10)
        amp_mu  = amp.mean(1, keepdim=True)
        amp_sig = amp.std(1, keepdim=True) + 1e-10
        amp_n   = (amp - amp_mu) / amp_sig

        f_amp_mu   = amp_mu                                     # 1  均值
        f_amp_sig  = amp_sig                                     # 2  标准差
        f_skew     = amp_n.pow(3).mean(1, keepdim=True)          # 3  偏度
        f_kurt     = amp_n.pow(4).mean(1, keepdim=True)          # 4  峰度 (关键!)
        f_papr     = amp.max(1, keepdim=True)[0] / (amp_mu + 1e-10)  # 5  峰均比

        # 幅度分位数 — 刻画 CDF 形状
        amp_sorted = amp.sort(1)[0]
        qi = [int(q * (L - 1)) for q in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)]
        f_quantiles = torch.stack([amp_sorted[:, i] for i in qi], dim=1)  # 6-11 (6维)

        # ══════ 高阶矩比 (经典 AMR 特征) ══════
        a2 = amp.pow(2).mean(1, keepdim=True)
        a4 = amp.pow(4).mean(1, keepdim=True)
        f_sigma_a = a4 / (a2.pow(2) + 1e-10)                    # 12  四阶矩比

        # ══════ 相位统计 ══════
        phase = torch.atan2(Q, I)
        dp = phase[:, 1:] - phase[:, :-1]
        dp = torch.atan2(torch.sin(dp), torch.cos(dp))          # wrap to [-π, π]
        f_phase_std  = dp.std(1, keepdim=True)                   # 13
        f_phase_mean = dp.abs().mean(1, keepdim=True)            # 14

        # ══════ IQ 统计 ══════
        Ic = I - I.mean(1, keepdim=True)
        Qc = Q - Q.mean(1, keepdim=True)
        I_std = Ic.std(1, keepdim=True) + 1e-10
        Q_std = Qc.std(1, keepdim=True) + 1e-10
        f_iq_corr = (Ic * Qc).mean(1, keepdim=True) / (I_std * Q_std)  # 15  IQ相关

        # ══════ 幅度分布形态 ══════
        f_above = (amp > amp_mu).float().mean(1, keepdim=True)  # 16  > 均值比例
        # 幅度 entropy 近似 (用排序差分的标准差)
        amp_diff = amp_sorted[:, 1:] - amp_sorted[:, :-1]
        f_entropy = amp_diff.std(1, keepdim=True)               # 17  分布均匀性

        # ══════ 六阶矩 (进一步区分高阶QAM) ══════
        a6 = amp.pow(6).mean(1, keepdim=True)
        f_sigma_6 = a6 / (a2.pow(3) + 1e-10)                    # 18

        # ══════ IQ 四阶矩 ══════
        f_I_kurt = ((Ic / I_std).pow(4)).mean(1, keepdim=True)  # 19
        f_Q_kurt = ((Qc / Q_std).pow(4)).mean(1, keepdim=True)  # 20

        return torch.cat([
            f_amp_mu, f_amp_sig, f_skew, f_kurt, f_papr,        # 5
            f_quantiles,                                          # 6
            f_sigma_a,                                            # 1
            f_phase_std, f_phase_mean,                            # 2
            f_iq_corr,                                            # 1
            f_above, f_entropy,                                   # 2
            f_sigma_6,                                            # 1
            f_I_kurt, f_Q_kurt,                                   # 2
        ], dim=1)  # (B, 20)