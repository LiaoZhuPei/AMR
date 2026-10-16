% 1. 加载并合成复信号
load('E:\XieYiR\UAV_osr_X\twin_droneRFa\twin_droneRF\train\A1_IN_S0_slice_1.mat');
x = complex(I, Q);  % 合成为复基带信号 x(t) = I(t) + j*Q(t)

% 2. 参数设置
fs = 80e6;          % 采样率 80 MHz
nfft = 2048;        % FFT 点数，决定时频分辨率平衡（2048点对应频率分辨率约 39 kHz）
window = hann(nfft);% 汉宁窗，抑制频谱泄漏
noverlap = round(nfft * 0.5); % 50% 重叠

% 3. 直接调用 spectrogram 绘图（显示基带 -40 MHz ~ +40 MHz）
figure;
spectrogram(x, window, noverlap, nfft, fs, 'centered', 'yaxis');
title('50 ms 全信号时频图 (基带)');
xlabel('时间 (s)');
ylabel('频率 (MHz)');
colorbar;