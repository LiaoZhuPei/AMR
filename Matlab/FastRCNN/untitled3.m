%% 1. 生成仿真信号 (模拟背景噪声 + 两个不同频段的突发信号)
clc; clear; close all;
fs = 1000;              % 采样率 1000Hz
T = 2;                  % 总时长 2秒
t = 0:1/fs:T-1/fs;
noise = 0.5 * randn(size(t)); % 高斯白噪声

% 信号A: 50Hz, 0.5s - 1.0s
sigA = 1.2 * sin(2*pi*50*t) .* (t >= 0.5 & t <= 1.0);
% 信号B: 线性调频(Chirp) 200Hz-300Hz, 1.2s - 1.8s
sigB = 1.5 * chirp(t-1.2, 200, 0.6, 300) .* (t >= 1.2 & t <= 1.8);

x = noise + sigA + sigB; % 混合信号

%% 2. 时频分析 (Spectrogram)
window = 128;
noverlap = 120;
nfft = 256;
[S, F, T_vec, P] = spectrogram(x, window, noverlap, nfft, fs);
P_db = 10*log10(abs(P)); % 转换为dB刻度

%% 3. 图像处理算法核心 (检测逻辑)
% A. 噪声底噪估计 (取中位数作为背景噪声基准)
noise_floor = median(P_db(:));
threshold = noise_floor + 12; % 设定阈值：比底噪高12dB

% B. 二值化 (生成黑白掩码)
binary_mask = P_db > threshold;

% C. 形态学去噪 (关键步骤：去除零星噪点，连接断裂的信号)
% 1. 去除小于 30 个像素的小噪点
binary_mask = bwareaopen(binary_mask, 30);
% 2. 闭运算：如果信号中间有小断裂，将其连接
se = strel('rectangle', [3, 3]);
binary_mask = imclose(binary_mask, se);

% D. 连通域提取 (RegionProps) -> 获取位置、频率、时长
stats = regionprops(binary_mask, 'BoundingBox', 'Area');

%% 4. 结果可视化与参数输出
figure('Position', [100, 100, 1000, 400]);
subplot(1, 2, 1);
imagesc(T_vec, F, P_db); axis xy; colormap jet; colorbar;
title('原始时频图'); xlabel('时间 (s)'); ylabel('频率 (Hz)');

subplot(1, 2, 2);
imagesc(T_vec, F, binary_mask); axis xy; colormap gray; hold on;
title('检测结果 (自动框选)'); xlabel('时间 (s)'); ylabel('频率 (Hz)');

fprintf('=== 检测报告 ===\n');
for k = 1:length(stats)
    % 获取边界框 [x, y, w, h] -> [时间索引, 频率索引, 时间宽度, 频率高度]
    bb = stats(k).BoundingBox;
    
    % 将像素坐标转换为物理单位
    % 注意：BoundingBox是 [col_start, row_start, width, height]
    t_idx_start = round(bb(1));
    f_idx_start = round(bb(2));
    t_len = bb(3);
    f_len = bb(4);
    
    % 防止索引越界
    t_idx_start = max(1, t_idx_start);
    f_idx_start = max(1, f_idx_start);
    
    % 计算物理值
    time_start = T_vec(t_idx_start);
    time_end = T_vec(min(length(T_vec), t_idx_start + round(t_len)));
    duration = time_end - time_start;
    
    freq_min = F(f_idx_start);
    freq_max = F(min(length(F), f_idx_start + round(f_len)));
    
    % 绘图框
    rectangle('Position', bb, 'EdgeColor', 'r', 'LineWidth', 2);
    text(bb(1), bb(2)+bb(4), sprintf('Signal %d', k), 'Color', 'r', 'FontWeight', 'bold');
    
    % 打印数据
    fprintf('信号 %d:\n', k);
    fprintf('  位置: %.3f 秒 -> %.3f 秒 (时长: %.3f s)\n', time_start, time_end, duration);
    fprintf('  频率: %.1f Hz -> %.1f Hz\n\n', freq_min, freq_max);
end