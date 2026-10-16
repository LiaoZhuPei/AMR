clc;
clear;
close all;

%% 1. 加载并合成复信号
load('E:\XieYiR\UAV_osr_X\twin_droneRFa\twin_droneRF\train\G1_IN_S1_slice_21.mat');
x = complex(I, Q);

%% 2. 参数设置
fs = 80e6;                   % 采样率 80 MHz
fc = 2.44e9;                 % 射频中心频率 2.44 GHz
nfft = 1024;                 % FFT 点数 2048
window = hann(nfft);         % 汉宁窗
noverlap = round(nfft * 0.5);% 50% 重叠 (1024 点)
N_slices = 100;               % 记录前 N 次频率转换的结果

%% 3. 计算 STFT 矩阵
[s, f, t] = spectrogram(x, window, noverlap, nfft, fs, 'centered');

%% 4. 坐标转换与幅值转 dB
f_rf_GHz = (f + fc) / 1e9;   % 换算为射频频率 (GHz)
t_ms = t * 1e3;              % 换算为毫秒 (ms)
P_dB = 20 * log10(abs(s));   % 幅度谱 (dB)

%% =================== 【功能 1 & 3：统计与尺寸打印】 ===================
% 1. 频域转换次数统计
num_transforms = size(s, 2); 
single_transform_size = size(s(:, 1));

fprintf('-------------------- 时频转换统计信息 --------------------\n');
fprintf('1. 频域转换总次数（时域滑动窗口数）：%d 次\n', num_transforms);
fprintf('2. 单次频域转换后的矩阵大小（行 x 列）：[%d, %d]\n', single_transform_size(1), single_transform_size(2));
fprintf('3. 完整时频矩阵 s 的大小/形状：[%d, %d] (共 %d 个频点 x %d 个时间切片)\n', ...
        size(s, 1), size(s, 2), size(s, 1), size(s, 2));

%% =================== 【功能 2：保存前 N 次单窗口转换结果】 ===================
sub_dir = 'first_N_window_spectra';
if ~exist(sub_dir, 'dir')
    mkdir(sub_dir);
end

fprintf('正在保存前 %d 次频率转换的结果至文件夹: %s ...\n', N_slices, sub_dir);

% 遍历前 N 次转换，保存单次窗口的瞬时幅度频谱
fig_single = figure('Visible', 'off'); 
for k = 1:min(N_slices, num_transforms)
    clf(fig_single);
    plot(f_rf_GHz, P_dB(:, k), 'b-', 'LineWidth', 1);
    grid on;
    xlim([min(f_rf_GHz), max(f_rf_GHz)]);
    ylim([min(P_dB(:)), max(P_dB(:))]);
    xlabel('射频频率 (GHz)');
    ylabel('幅度 (dB)');
    title(sprintf('第 %d 次频域转换 (时间 t = %.4f ms)', k, t_ms(k)));
    
    single_save_name = fullfile(sub_dir, sprintf('window_spectrum_%02d.png', k));
    exportgraphics(fig_single, single_save_name, 'Resolution', 150);
end
close(fig_single);
fprintf('前 %d 次频率转换的单窗口频谱图已保存完毕。\n', N_slices);

%% 5. 绘制并保存完整时频图
fig = figure('Visible', 'on'); 
imagesc(t_ms, f_rf_GHz, P_dB);
axis xy;
colormap('turbo');
c = colorbar;
c.Label.String = '幅度 (dB)';
clim([max(P_dB(:)) - 50, max(P_dB(:))]); % 动态范围，过滤底噪

xlabel('时间 (ms)');
ylabel('射频频率 (GHz)');
title('50 ms 信号射频时频图 (中心频点 2.44 GHz)');

% 保存主时频图
save_filename = 'G1_IN_S1_slice_21.png';
exportgraphics(fig, save_filename, 'Resolution', 300);
fprintf('完整时频图已成功保存为：%s\n', save_filename);

%% =================== 【功能 3 补充：获取并打印图片分辨率】 ===================
img_info = imfinfo(save_filename);
fprintf('4. 保存的图片实际像素大小：宽度 %d px × 高度 %d px\n', img_info.Width, img_info.Height);
fprintf('----------------------------------------------------------\n');