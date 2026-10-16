%% ==================== 1. 参数自由配置区 ====================
clear; clc; close all;

% --- 文件与系统参数 ---
mat_file    = 'E:\XieYiR\UAV_osr_X\twin_droneRFa\twin_droneRF\train\A1_IN_S0_slice_1.mat'; % 目标数据文件
fs          = 80e6;                   % 采样率 80 MHz
fc          = 2.44e9;                 % 射频中心频率 2.44 GHz

% --- 信号切分配置 ---
seg_len     = 131328;                   % 每个切片的点数 (默认 1024 点，对应 12.8 μs)
seg_overlap = 0;                      % 切片重叠点数 (默认 0 点)

% --- 绘图与保存配置 ---
num_to_plot = 10;                     % 绘制并保存前 N 个片段的时频图 (默认 10 个)
img_save_dir = './spectrogram_results';% 时频图输出目录

% --- STFT 时频分析参数 (适应切片长度) ---
% 当 seg_len=1024 时，窗长不能超过 1024。推荐取 128 或 256
stft_win_len = min(512, seg_len);     % STFT 窗长
stft_nfft    = 512;                   % STFT FFT点数 (大于等于窗长以提高频域插值平滑度)
stft_overlap = round(stft_win_len * 0.5); % STFT 滑窗重叠率 (50%)

%% ==================== 2. 数据加载与校验 ====================
if ~isfile(mat_file)
    error('未找到数据文件：%s，请检查路径。', mat_file);
end

fprintf('正在加载数据：%s ...\n', mat_file);
load(mat_file);
x = complex(I, Q);
total_len = length(x);

% 检查参数合法性
assert(seg_overlap < seg_len, '切片重叠点数必须严格小于切片长度！');

% 计算最大可切分片段数与步长
step = seg_len - seg_overlap;
total_segments = floor((total_len - seg_overlap) / step);

fprintf('数据总点数：%d，切分长度：%d，步长：%d\n', total_len, seg_len, step);
fprintf('共计可切分为 %d 个片段，将处理并保存前 %d 个片段的时频图。\n', ...
        total_segments, min(num_to_plot, total_segments));

if ~exist(img_save_dir, 'dir')
    mkdir(img_save_dir);
end

%% ==================== 3. 切片数据保存 (当前版本已注释) ====================
%{
% -------------------------------------------------------------
% 【保留备用】若后续需要将切片后的纯 IQ 数据分别存成 .mat 文件，请取消该段注释
% -------------------------------------------------------------
slice_data_dir = './sliced_mat_data';
if ~exist(slice_data_dir, 'dir'), mkdir(slice_data_dir); end

for k = 1:total_segments
    start_idx = (k - 1) * step + 1;
    end_idx   = start_idx + seg_len - 1;
    
    I_seg = I(start_idx:end_idx);
    Q_seg = Q(start_idx:end_idx);
    
    seg_filename = fullfile(slice_data_dir, sprintf('seg_%05d.mat', k));
    save(seg_filename, 'I_seg', 'Q_seg', 'fs', 'fc');
end
fprintf('切片数据保存完成！\n');
%}

%% ==================== 4. 循环绘制并保存前 N 个时频图 ====================
win = hann(stft_win_len);
num_process = min(num_to_plot, total_segments);

% 使用隐藏窗口批量绘图，防止连续弹出 10 个窗口抢占屏幕
fig = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);

for i = 1:num_process
    % 计算当前切片的起始和终止索引
    start_idx = (i - 1) * step + 1;
    end_idx   = start_idx + seg_len - 1;
    x_seg     = x(start_idx:end_idx);
    
    % 计算 STFT
    [s, f, t] = spectrogram(x_seg, win, stft_overlap, stft_nfft, fs, 'centered');
    
    % 转换轴坐标
    f_rf_GHz = (f + fc) / 1e9;    % 射频频率 (GHz)
    t_us     = t * 1e6;           % 片段内局部时间 (微秒 μs)
    P_dB     = 20 * log10(abs(s) + eps); % 幅度谱转 dB (加 eps 防止 log(0))
    
    % 绘制时频图像
    clf(fig);
    imagesc(t_us, f_rf_GHz, P_dB);
    axis xy;
    colormap(fig, 'turbo');
    c = colorbar;
    c.Label.String = '幅度 (dB)';
    clim([max(P_dB(:)) - 50, max(P_dB(:))]); % 显示峰值下 50 dB 动态范围
    
    % 计算当前片段在整个 50ms 信号中的全局起始时间
    global_start_time_ms = (start_idx - 1) / fs * 1e3;
    
    xlabel('切片内相对时间 (\mus)');
    ylabel('射频频率 (GHz)');
    title(sprintf('片段 #%d 时频图 (全局起点: %.4f ms, 长度: %d 点)', ...
                  i, global_start_time_ms, seg_len));
    
    % 保存图片 (300 DPI 高清输出)
    img_name = fullfile(img_save_dir, sprintf('spectrogram_seg_%03d.png', i));
    exportgraphics(fig, img_name, 'Resolution', 300);
    
    fprintf('已保存时频图 [%02d/%02d] -> %s\n', i, num_process, img_name);
end

close(fig); % 关闭后台绘图句柄
fprintf('\n所有指定时频图生成完毕！图片保存在目录：%s\n', img_save_dir);