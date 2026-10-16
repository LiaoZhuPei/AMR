"""
Ultimate TCN for AMR (自动调制识别终极版)
包含: DSP前馈预处理 (A通道, 差分相位, FFT频谱) + 双流网络 (时域TCN + 频域CNN) + 双池化 + 统计特征补偿
"""
import torch
import torch.nn as nn
import torch.fft


# =====================================================================
# 1. DSP 物理预处理层 (无训练参数，利用 PyTorch 张量加速物理特征提取)
# =====================================================================
class DSP_FrontEnd(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (batch, 1024, 2)
        I = x[:, :, 0]
        Q = x[:, :, 1]

        # 1. 计算瞬时幅度 A (使用 1e-4 防止 float16 AMP 混合精度下梯度爆炸)
        A = torch.sqrt(I ** 2 + Q ** 2 + 1e-4)

        # 2. 计算差分相位 (解卷绕处理，消除恒定频偏)
        phase = torch.atan2(Q, I)
        diff_phase = phase[:, 1:] - phase[:, :-1]
        diff_phase = torch.cat([torch.zeros_like(diff_phase[:, :1]), diff_phase], dim=1)  # 补齐长度
        diff_phase = torch.remainder(diff_phase + torch.pi, 2 * torch.pi) - torch.pi

        # 3. 计算 FFT 频谱幅度 (针对 FSK)
        complex_sig = torch.complex(I, Q)
        fft_val = torch.fft.fft(complex_sig, dim=1)
        fft_val = torch.fft.fftshift(fft_val, dim=1)
        fft_amp = torch.log10(torch.abs(fft_val) + 1e-4)  # 对数幅度

        # 时域特征拼接 (batch, 4, 1024) -> I, Q, A, Diff_Phase
        time_features = torch.stack([I, Q, A, diff_phase], dim=1)

        # 频域特征 (batch, 1, 1024)
        freq_features = fft_amp.unsqueeze(1)

        return time_features, freq_features, A, I, Q


# =====================================================================
# 2. 基础组件 (CNN 前端 & TCN 块 & 分类头)
# =====================================================================
class CNNFrontend(nn.Module):
    def __init__(self, in_channels=4, out_channels=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv1d(64, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.cnn(x)


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),  # 加宽一点以容纳多维度特征
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)


# =====================================================================
# 3. 终极主网络组合
# =====================================================================
class Ultimate_TCN(nn.Module):
    def __init__(self, num_classes=14, tcn_channels=128, num_layers=7, dropout=0.2):
        super().__init__()

        self.dsp = DSP_FrontEnd()

        # --- 1. 时域流 (Time Stream) ---
        self.time_cnn = CNNFrontend(in_channels=4, out_channels=tcn_channels)
        tcn_blocks = []
        for i in range(num_layers):
            dilation = 2 ** i
            tcn_blocks.append(TCNBlock(tcn_channels, 3, dilation, dropout))
        self.tcn = nn.Sequential(*tcn_blocks)
        self.time_norm = nn.BatchNorm1d(tcn_channels)

        # --- 2. 频域流 (Freq Stream) ---
        # 简单高效地将 1024 长度频谱压缩为 64 维特征向量
        self.freq_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # 压扁为 (batch, 64, 1)
        )

        # --- 3. 统计特征流 (Stats Stream) ---
        self.num_stats = 5
        self.stats_bn = nn.BatchNorm1d(self.num_stats)  # 必须加 BN 防止特征尺度失衡

        # --- 4. 分类器 (维度 = 时域双池化(128*2) + 频域(64) + 统计(5) = 325) ---
        fused_dim = tcn_channels * 2 + 64 + self.num_stats
        self.classifier = ClassifierHead(fused_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, 1024, 2)

        # 1. 物理预处理
        time_feat, freq_feat, A, I, Q = self.dsp(x)

        # 2. 提取统计特征 (针对 QAM)
        A_mean = A.mean(dim=1, keepdim=True)
        A_std = A.std(dim=1, keepdim=True)
        A_max = A.max(dim=1, keepdim=True)[0]
        A_min = A.min(dim=1, keepdim=True)[0]

        P = I ** 2 + Q ** 2
        PAPR = P.max(dim=1, keepdim=True)[0] / (P.mean(dim=1, keepdim=True) + 1e-4)

        stats_feat = torch.cat([A_mean, A_std, A_max, A_min, PAPR], dim=1)
        stats_feat = self.stats_bn(stats_feat)  # 归一化统计量

        # 3. 跑时域流 (Time Stream)
        t_out = self.time_cnn(time_feat)
        t_out = self.tcn(t_out)
        t_out = self.time_norm(t_out)
        # 双池化 (Dual Pooling: Mean + Max)
        t_mean = t_out.mean(dim=2)  # (batch, 128)
        t_max = t_out.max(dim=2)[0]  # (batch, 128)
        t_pooled = torch.cat([t_mean, t_max], dim=1)  # (batch, 256)

        # 4. 跑频域流 (Freq Stream)
        f_out = self.freq_cnn(freq_feat).squeeze(-1)  # (batch, 64)

        # 5. 终极融合并分类
        fused = torch.cat([t_pooled, f_out, stats_feat], dim=1)  # (batch, 325)

        return self.classifier(fused)


def build_model(classes=14):
    return Ultimate_TCN(num_classes=classes)