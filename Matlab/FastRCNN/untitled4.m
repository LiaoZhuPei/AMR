%% === 1. 用户配置区 (请务必修改这里) ===
clc; clear; close all;

% --- A. 文件加载设置 ---
% 假设你的 mat 文件名是 'my_iq_data.mat'
filename = 'E:\M_Project\FasteRCNNdata\RFFrameDate\SpectrogramRFFrameDetectionData\frame_138769090412766230_bw_25E+6.mat'; 

% --- B. 关键参数设置 (非常重要！) ---
fs = 1e6;       % 采样率 (Hz)，例如 1MHz。必须设置正确，否则结果单位不对！
win_len = 1024; % 窗口长度。如果是窄带信号建议设大(2048)，宽带突发信号设小(256)
dB_offset = 10; % 灵敏度阈值：比底噪高多少dB算信号？(建议 8~15)

%% === 2. 数据加载与预处理 ===
% 为了演示，如果你没有文件，我这里先生成一段模拟的 IQ 信号
% 【实际使用时，请注释掉下面这段 "生成模拟数据" 的代码】
% ---------------------------------------------------------
if ~isfile(filename)
    fprintf('未找到文件，正在生成模拟 IQ 信号...\n');
    N = 710000;
    t_sim = (0:N-1)/fs;
    noise = (randn(1,N) + 1j*randn(1,N)) * 0.1; % 复高斯白噪声
    
    % 模拟信号1: 正频率的一段信号 (0.1s - 0.2s, +100kHz)
    sig1 = 0.5 * exp(1j*2*pi*100e3*t_sim) .* (t_sim > 0.1 & t_sim < 0.2);
    
    % 模拟信号2: 负频率的一段线性调频 (-200kHz 到 -100kHz)
    sig2 = 0.8 * exp(1j*2*pi*(-200e3*t_sim + (100e3/0.4)*t_sim.^2/2)) .* (t_sim > 0.4 & t_sim < 0.6);
    
    iq_data = single(noise + sig1 + sig2); % 模拟你的 single 格式
else
    % 【实际加载逻辑】
    dataStruct = load(filename);
    vars = fieldnames(dataStruct);
    % 自动寻找那个 1x710000 的变量
    iq_data = dataStruct.(vars{1}); 
    fprintf('已加载变量: %s, 大小: %d x %d\n', vars{1}, size(iq_data,1), size(iq_data,2));
end
% ---------------------------------------------------------

% 确保数据是双精度 (double) 以避免计算误差，并转为行向量
x = double(reshape(iq_data, 1, []));

%% === 3. 核心算法：针对复数信号的时频分析 ===
fprintf('正在计算 STFT...\n');
noverlap = round(win_len * 0.75); % 75% 重叠，图像更细腻
nfft = win_len;

% [关键] 使用 'centered' 参数，频率轴范围自动变为 [-fs/2, fs/2]
[S, F, T, P] = spectrogram(x, win_len, noverlap, nfft, fs, 'centered');

% 转换为 dB 功率谱
P_db = 10*log10(abs(P)); 

%% === 4. 图像处理与信号提取 ===
fprintf('正在进行检测...\n');

% A. 自适应阈值计算
% 使用中位数作为底噪基准
noise_floor = median(P_db(:));
threshold = noise_floor + dB_offset;

% B. 二值化与形态学去噪
binary_mask = P_db > threshold;

% 去除孤立的小噪点 (面积小于 50 像素)
binary_mask = bwareaopen(binary_mask, 50);

% 闭运算：将断裂的信号连接起来 (纵向频率方向，横向时间方向)
se = strel('rectangle', [5, 5]); 
binary_mask = imclose(binary_mask, se);

% C. 提取连通区域 (RegionProps)
stats = regionprops(binary_mask, 'BoundingBox', 'Area');

%% === 5. 结果可视化与数据输出 (修复标注版) ===
figure('Position', [100, 100, 1200, 700]);

% --- 上图：原始时频图 ---
subplot(2, 1, 1);
imagesc(T, F, P_db); axis xy; colormap jet; colorbar;
caxis([noise_floor-5, noise_floor+25]); 
title('原始 IQ 信号时频图');
ylabel('频率 (Hz)'); xlabel('时间 (s)');

% --- 下图：彩色检测结果 + 详细标注 ---
subplot(2, 1, 2);

% 1. 生成标记矩阵和彩色底图
[L, num_found] = bwlabel(binary_mask);
rgb_image = label2rgb(L, 'hsv', 'k', 'shuffle'); % 信号彩色，背景黑

% 2. 绘制彩色底图 (使用物理坐标轴 T 和 F)
imagesc(T, F, rgb_image); axis xy; hold on;
title(sprintf('检测结果 (共发现 %d 个目标)', num_found));
ylabel('频率 (Hz)'); xlabel('时间 (s)');

fprintf('\n=== 检测结果详细报告 ===\n');
fprintf('底噪估计: %.2f dB | 检测阈值: %.2f dB\n', noise_floor, threshold);
fprintf('------------------------------------------------------------------------------------------\n');
fprintf('%-4s | %-10s | %-10s | %-24s | %-10s\n', ...
    'ID', '起始时间(s)', '时长(s)', '频率范围 (Hz)', '带宽 (Hz)');
fprintf('------------------------------------------------------------------------------------------\n');

for k = 1:num_found
    % 获取当前 ID 的像素边界 [x_pix, y_pix, w_pix, h_pix]
    % 注意：regionprops 是基于矩阵索引的
    stats_k = regionprops(L == k, 'BoundingBox');
    bb_pix = stats_k.BoundingBox;
    
    % === [核心修复] 将像素坐标转换为物理坐标 ===
    
    % 1. 提取像素索引 (四舍五入并防止越界)
    idx_t_start = max(1, round(bb_pix(1)));
    idx_f_start = max(1, round(bb_pix(2)));
    
    idx_t_end = min(length(T), round(bb_pix(1) + bb_pix(3)));
    idx_f_end = min(length(F), round(bb_pix(2) + bb_pix(4)));
    
    % 2. 查表得到物理数值 (秒, 赫兹)
    val_t_start = T(idx_t_start);
    val_t_end   = T(idx_t_end);
    val_f_start = F(idx_f_start);
    val_f_end   = F(idx_f_end);
    
    % 3. 计算物理宽高
    phys_width  = val_t_end - val_t_start;
    phys_height = val_f_end - val_f_start;
    
    % 4. 构造物理坐标系的矩形位置 [x, y, w, h]
    rect_pos = [val_t_start, val_f_start, phys_width, phys_height];
    
    % === 绘图标注 ===
    
    % A. 画框 (亮黄色，加粗，保证对比度)
    rectangle('Position', rect_pos, 'EdgeColor', 'y', 'LineWidth', 2, 'LineStyle', '-');
    
    % B. 标 ID (在框的左上角)
    % BackgroundColor 设为黑色，防止文字被信号颜色淹没
    text(val_t_start, val_f_end, sprintf(' ID:%d ', k), ...
        'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', 'k', 'Margin', 1, ...
        'VerticalAlignment', 'bottom');
    
    % C. 标带宽 (在框的内部下方)
    text(val_t_start, val_f_start, sprintf(' BW:%.0fHz ', phys_height), ...
        'Color', 'w', 'FontSize', 9, ...
        'VerticalAlignment', 'bottom');
    
    % === 打印数据 ===
    freq_range_str = sprintf('%.0f ~ %.0f', val_f_start, val_f_end);
    fprintf('#%-3d | %-10.4f | %-10.4f | %-24s | %-10.0f\n', ...
        k, val_t_start, phys_width, freq_range_str, phys_height);
end
fprintf('------------------------------------------------------------------------------------------\n');