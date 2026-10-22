# features/extractor.py
import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from config import Config

class FeatureExtractor:
    """
    从复数IQ信号提取多域特征
    特征维度说明：
    - 频谱特征: 频谱均值/方差/峰度/偏度/峰值频率/谱熵/谱平坦度/谱质心
    - 瞬时特征: 瞬时幅度/频率/相位的统计量
    - 循环平稳特征: 循环自相关函数峰值
    - 高阶统计量: 二阶/四阶累积量
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fs = cfg.FS
        self.n_fft = cfg.N_FFT
    
    # ------------------------------------------------------------------ #
    #                         频谱特征                                      #
    # ------------------------------------------------------------------ #
    def _spectral_features(self, iq: np.ndarray) -> np.ndarray:
        """基于功率谱密度的特征"""
        # Welch方法估计PSD
        freqs, psd = scipy_signal.welch(
            iq, fs=self.fs, 
            nperseg=self.n_fft, 
            return_onesided=False
        )
        psd = np.abs(psd)
        psd_norm = psd / (psd.sum() + 1e-12)  # 归一化为概率分布
        
        # 谱统计量
        mean_psd   = np.mean(psd)
        std_psd    = np.std(psd)
        max_psd    = np.max(psd)
        peak_freq  = freqs[np.argmax(psd)]
        
        # 谱熵（Shannon）
        spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
        
        # 谱平坦度（Wiener entropy）
        geo_mean = np.exp(np.mean(np.log(psd + 1e-12)))
        ari_mean = np.mean(psd) + 1e-12
        spec_flatness = geo_mean / ari_mean
        
        # 谱质心
        spec_centroid = np.sum(freqs * psd_norm)
        
        # 谱带宽（加权标准差）
        spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * psd_norm))
        
        # 谱峰度/偏度
        spec_kurtosis = kurtosis(psd)
        spec_skewness = skew(psd)
        
        # 峰均比
        papr = max_psd / (mean_psd + 1e-12)
        
        features = np.array([
            mean_psd, std_psd, max_psd, peak_freq,
            spec_entropy, spec_flatness, spec_centroid,
            spec_bandwidth, spec_kurtosis, spec_skewness, papr
        ])
        return features  # dim=11
    
    # ------------------------------------------------------------------ #
    #                       瞬时特征                                        #
    # ------------------------------------------------------------------ #
    def _instantaneous_features(self, iq: np.ndarray) -> np.ndarray:
        """瞬时幅度、频率、相位特征"""
        # 瞬时幅度
        inst_amp = np.abs(iq)
        
        # 瞬时相位（解卷绕）
        inst_phase = np.unwrap(np.angle(iq))
        
        # 瞬时频率（相位差分）
        inst_freq = np.diff(inst_phase) * self.fs / (2 * np.pi)
        
        def stat_features(x):
            return np.array([
                np.mean(x), np.std(x),
                kurtosis(x), skew(x),
                np.percentile(x, 25),
                np.percentile(x, 75),
                np.max(x) - np.min(x)   # 动态范围
            ])
        
        amp_feats   = stat_features(inst_amp)    # dim=7
        phase_feats = stat_features(inst_phase)  # dim=7
        freq_feats  = stat_features(inst_freq)   # dim=7
        
        # 幅度归一化方差（AM成分指标）
        amp_var_norm = np.var(inst_amp) / (np.mean(inst_amp) ** 2 + 1e-12)
        
        # 零交叉率（频率粗估计）
        centered = inst_freq - np.mean(inst_freq)
        zcr = np.sum(np.diff(np.sign(centered)) != 0) / len(centered)
        
        extra = np.array([amp_var_norm, zcr])  # dim=2
        
        return np.concatenate([amp_feats, phase_feats, freq_feats, extra])  # dim=23
    
    # ------------------------------------------------------------------ #
    #                     高阶统计量（HOS）                                 #
    # ------------------------------------------------------------------ #
    def _higher_order_statistics(self, iq: np.ndarray) -> np.ndarray:
        """
        高阶累积量特征，对调制识别有效
        参考: Swami & Sadler (2000) - Hierarchical digital modulation classification
        """
        # 归一化信号
        sigma = np.sqrt(np.mean(np.abs(iq) ** 2))
        x = iq / (sigma + 1e-12)
        
        # 二阶累积量
        C20 = np.mean(x ** 2)                     # E[x^2]
        C21 = np.mean(np.abs(x) ** 2)             # E[|x|^2] = 1 after norm
        
        # 四阶累积量
        C40 = np.mean(x ** 4) - 3 * (np.mean(x ** 2)) ** 2
        C41 = np.mean(np.abs(x) ** 2 * x ** 2) - \
              2 * np.mean(x ** 2) * np.mean(np.abs(x) ** 2) - \
              np.abs(np.mean(x ** 2)) ** 2
        C42 = np.mean(np.abs(x) ** 4) - \
              np.abs(np.mean(x ** 2)) ** 2 - \
              2 * (np.mean(np.abs(x) ** 2)) ** 2
        
        # 六阶累积量（简化）
        C60 = np.mean(x ** 6) - 15 * np.mean(x ** 4) * np.mean(x ** 2) + \
              30 * (np.mean(x ** 2)) ** 3
        
        features = np.array([
            np.abs(C20), np.abs(C21),
            np.abs(C40), np.abs(C41), np.abs(C42),
            np.abs(C60)
        ])
        return features  # dim=6
    
    # ------------------------------------------------------------------ #
    #                     循环平稳特征                                      #
    # ------------------------------------------------------------------ #
    def _cyclostationary_features(self, iq: np.ndarray, 
                                   n_alpha: int = 8) -> np.ndarray:
        """
        循环自相关函数（CAF）特征
        通过检测循环频率alpha处的谱相关，识别周期性调制特征
        """
        N = len(iq)
        # 检测循环频率范围
        alphas = np.linspace(0, self.fs / 4, n_alpha + 1)[1:]  # 避免alpha=0
        
        caf_peaks = []
        for alpha in alphas:
            # 计算循环自相关 R_x^alpha(0)
            shifts = np.arange(0, min(64, N // 4))
            caf_vals = []
            for lag in shifts[:8]:  # 取8个lag值
                if lag == 0:
                    term = np.mean(iq * np.conj(iq) * 
                                   np.exp(-1j * 2 * np.pi * alpha * 
                                          np.arange(N) / self.fs))
                else:
                    term = np.mean(iq[lag:] * np.conj(iq[:-lag]) * 
                                   np.exp(-1j * 2 * np.pi * alpha * 
                                          np.arange(N - lag) / self.fs))
                caf_vals.append(np.abs(term))
            caf_peaks.append(np.max(caf_vals))
        
        features = np.array(caf_peaks)  # dim=n_alpha
        return features
    
    # ------------------------------------------------------------------ #
    #                     自相关特征                                        #
    # ------------------------------------------------------------------ #
    def _autocorrelation_features(self, iq: np.ndarray, 
                                   max_lag: int = 64) -> np.ndarray:
        """时域自相关特征，反映信号的周期性"""
        # 归一化自相关
        acf = np.correlate(iq, iq, mode='full')
        acf = acf[len(acf) // 2:]          # 取正lag部分
        acf = np.abs(acf) / (np.abs(acf[0]) + 1e-12)
        acf = acf[1:max_lag + 1]           # 排除lag=0
        
        # 统计特征
        peak_val   = np.max(acf)
        peak_lag   = np.argmax(acf) + 1
        mean_acf   = np.mean(acf)
        std_acf    = np.std(acf)
        
        # 衰减速率（线性拟合）
        lags = np.arange(1, len(acf) + 1)
        decay_rate = np.polyfit(lags, acf, 1)[0]
        
        # 过零点数（反映振荡特性）
        zcr_acf = np.sum(np.diff(np.sign(acf - 0.1)) != 0)
        
        features = np.array([peak_val, peak_lag, mean_acf, 
                              std_acf, decay_rate, zcr_acf])
        return features  # dim=6
    
    # ------------------------------------------------------------------ #
    #                    主提取函数                                         #
    # ------------------------------------------------------------------ #
    def extract(self, iq: np.ndarray) -> np.ndarray:
        """
        从单个IQ信号提取完整特征向量
        输入: iq (2048,) complex
        输出: feature_vec (N_features,) float
        """
        spec_feats   = self._spectral_features(iq)           # 11
        inst_feats   = self._instantaneous_features(iq)      # 23
        hos_feats    = self._higher_order_statistics(iq)     # 6
        cyc_feats    = self._cyclostationary_features(iq)    # 8
        acf_feats    = self._autocorrelation_features(iq)    # 6
        
        feature_vec = np.concatenate([
            spec_feats, inst_feats, hos_feats, cyc_feats, acf_feats
        ])  # 总维度 = 54
        
        return feature_vec
    
    def extract_batch(self, signals: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        批量提取特征
        输入: signals (N, 2048) complex
        输出: features (N, n_features) float
        """
        n = len(signals)
        features = []
        
        for i, sig in enumerate(signals):
            if verbose and (i % 500 == 0):
                print(f"  [特征提取] {i}/{n} ({100*i/n:.1f}%)")
            feat = self.extract(sig)
            features.append(feat)
        
        return np.array(features)