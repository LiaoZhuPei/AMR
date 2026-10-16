clc
clear
% 1. 加载并合成复信号
load('E:\XieYiR\UAV_osr_X\twin_droneRFa\twin_droneRF\train\G1_IN_S1_slice_21.mat');
x = complex(I, Q);

% 2. 参数设置
fs = 80e6;                  % 采样率 80 MHz
fc = 2.44e9;                % 射频中心频率 2.44 GHz
nfft = 2048;                % FFT 点数
window = hann(nfft);        % 汉宁窗
noverlap = round(nfft * 0.5); % 50% 重叠

% 3. 计算 STFT 矩阵
[s, f, t] = spectrogram(x, window, noverlap, nfft, fs, 'centered');

% 4. 坐标转换与幅值转 dB
f_rf_GHz = (f + fc) / 1e9;  % 换算为射频频率 (GHz)，范围为 2.40 ~ 2.48 GHz
t_ms = t * 1e3;             % 换算为毫秒 (ms)
P_dB = 20 * log10(abs(s));  % 幅度谱 (dB)

% 5. 绘制时频图
fig = figure('Visible', 'on'); % 若想后台批量画图不弹窗，可设为 'off'
imagesc(t_ms, f_rf_GHz, P_dB);
axis xy;
colormap('turbo');
c = colorbar;
c.Label.String = '幅度 (dB)';
clim([max(P_dB(:)) - 50, max(P_dB(:))]); % 动态范围，过滤底噪

xlabel('时间 (ms)');
ylabel('射频频率 (GHz)');
title('50 ms 信号射频时频图 (中心频点 2.44 GHz)');

% ==================== 【新增：保存图片步骤】 ====================
save_filename = 'G1_IN_S1_slice_21.png';

% 【推荐方式 1】适用于 MATLAB R2020a 及以上版本
% 优势：自动紧凑裁切白边、支持指定 300 DPI 高清输出
exportgraphics(fig, save_filename, 'Resolution', 300);

% 【备用方式 2】经典 saveas（通用性好，屏幕原生分辨率）
% saveas(fig, save_filename);

% 【备用方式 3】传统高清 print（适用于旧版 MATLAB）
% print(fig, save_filename, '-dpng', '-r300');

fprintf('时频图已成功保存为：%s\n', save_filename);