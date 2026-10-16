%% === 1. 系统初始化与数据加载 ===
clc; clear; close all;

% --- A. 用户设置 ---
filename = 'E:\M_Project\FasteRCNNdata\waitA\20170515_Part_3.mat'; 
total_duration = 10; % 总时长 10秒
fs = 71000;          % 采样率 (根据 710000点 / 10秒 计算得出的假设值，请修改)
window_size_sec = 1; % 单个窗口显示的长度 (1秒)

% 播放速度控制 (秒)
% 设为 0 表示全力全速跑，设为 0.5 表示每帧停留 0.5秒以便肉眼观察
refresh_delay = 0.5; 

% STFT 参数
stft_win = 512;      % 窗口大小
stft_overlap = 256;  % 重叠
stft_nfft = 512;     % FFT点数

% 检测参数
dB_offset = 12;      % 阈值灵敏度

% --- B. 数据准备 ---
% (如果没有文件，生成模拟数据)
if ~isfile(filename)
    fprintf('正在生成模拟信号 (10秒)...\n');
    N = fs * total_duration;
    t_full = (0:N-1)/fs;
    % 模拟背景噪声
    x_full = (randn(1,N) + 1j*randn(1,N)) * 0.1;
    
    % 在不同时间段植入不同信号
    % 0.5s: 信号A
    x_full(t_full>0.5 & t_full<0.8) = x_full(t_full>0.5 & t_full<0.8) + 1.0*exp(1j*2*pi*10e3*t_full(t_full>0.5 & t_full<0.8));
    % 2.5s: 信号B
    x_full(t_full>2.5 & t_full<2.9) = x_full(t_full>2.5 & t_full<2.9) + 1.2*exp(1j*2*pi*-20e3*t_full(t_full>2.5 & t_full<2.9));
    % 8.2s: 信号C
    x_full(t_full>8.2 & t_full<8.8) = x_full(t_full>8.2 & t_full<8.8) + 1.5*exp(1j*2*pi*5e3*t_full(t_full>8.2 & t_full<8.8));
    
    x_full = single(x_full);
else
    dataStruct = load(filename);
    vars = fieldnames(dataStruct);
    x_full = double(dataStruct.(vars{1}));
    % 确保是行向量
    if iscolumn(x_full), x_full = x_full'; end
end

%% === 2. 初始化绘图窗口 ===
hFig = figure('Position', [100, 100, 1000, 600], 'Name', '实时信号监测系统');

% 预计算分块信息
samples_per_window = floor(window_size_sec * fs);
num_windows = floor(length(x_full) / samples_per_window);

fprintf('开始实时监测...\n');
fprintf('总时长: %d s | 窗口: %d s | 总帧数: %d\n', total_duration, window_size_sec, num_windows);

%% === 3. 实时循环处理 ===
for i = 1:num_windows
    % A. 提取当前窗口数据
    idx_start = (i-1) * samples_per_window + 1;
    idx_end   = idx_start + samples_per_window - 1;
    
    x_chunk = x_full(idx_start:idx_end);
    
    % 当前物理时间偏移量 (例如第2秒开始)
    time_offset = (i-1) * window_size_sec;
    
    % B. 计算 STFT
    % 使用 'centered' 处理 IQ 信号负频率
    [S, F, T, P] = spectrogram(double(x_chunk), stft_win, stft_overlap, stft_nfft, fs, 'centered');
    
    % 将 T 轴加上偏移量，使其显示真实的物理时间 (如 2.1s, 2.2s...)
    T_phys = T + time_offset;
    
    P_db = 10*log10(abs(P));
    
    % C. 检测算法
    noise_floor = median(P_db(:));
    threshold = noise_floor + dB_offset;
    binary_mask = P_db > threshold;
    binary_mask = bwareaopen(binary_mask, 30); % 去噪
    se = strel('rectangle', [5, 5]);
    binary_mask = imclose(binary_mask, se);    % 连通
    
    % D. 生成彩色标记图
    [L, num_found] = bwlabel(binary_mask);
    rgb_image = label2rgb(L, 'hsv', 'k', 'shuffle');
    
    % --- E. 绘图刷新 (关键步骤) ---
    figure(hFig); % 激活窗口
    cla;          % 清除上一帧的内容
    
    % 1. 画彩色底图
    imagesc(T_phys, F, rgb_image); 
    axis xy; 
    
    % 2. 设置固定的坐标轴范围 (让画面不乱跳)
    xlim([time_offset, time_offset + window_size_sec]);
    ylim([-fs/2, fs/2]); 
    
    title(sprintf('实时监测窗口 [%.1f s - %.1f s] | 发现目标: %d', ...
        time_offset, time_offset+window_size_sec, num_found), 'FontSize', 14);
    xlabel('物理时间 (s)'); ylabel('频率 (Hz)');
    
    % 3. 遍历标注 (画框)
    hold on;
    for k = 1:num_found
        stats = regionprops(L==k, 'BoundingBox');
        bb_pix = stats.BoundingBox;
        
        % 坐标转换 (像素 -> 物理)
        % 注意边界限制
        t_idx_1 = max(1, round(bb_pix(1)));
        t_idx_2 = min(length(T), round(bb_pix(1) + bb_pix(3)));
        f_idx_1 = max(1, round(bb_pix(2)));
        f_idx_2 = min(length(F), round(bb_pix(2) + bb_pix(4)));
        
        t_phy_start = T_phys(t_idx_1);
        t_phy_end   = T_phys(t_idx_2);
        f_phy_start = F(f_idx_1);
        f_phy_end   = F(f_idx_2);
        
        w_phy = t_phy_end - t_phy_start;
        h_phy = f_phy_end - f_phy_start;
        
        % 画框
        rectangle('Position', [t_phy_start, f_phy_start, w_phy, h_phy], ...
            'EdgeColor', 'y', 'LineWidth', 2);
        
        % 标字
        text(t_phy_start, f_phy_end, sprintf('ID:%d\n%.1fkHz', k, (f_phy_start+f_phy_end)/2/1000), ...
            'Color', 'y', 'FontWeight', 'bold', 'FontSize', 10, ...
            'VerticalAlignment', 'bottom', 'BackgroundColor', 'k');
        
        % 控制台输出日志 (可选)
        fprintf('[Frame %d] 检测到 ID:%d | 时间: %.2fs | 频率: %.1f Hz\n', ...
            i, k, t_phy_start, (f_phy_start+f_phy_end)/2);
    end
    hold off;
    
    % F. 强制刷新与延时
    drawnow; % 立即更新画面
    
    if i < num_windows
        pause(refresh_delay); % 控制播放速度
    end
end

fprintf('监测结束。\n');