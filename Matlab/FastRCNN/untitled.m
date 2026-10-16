clear; close all; clc;

%% 1. 参数设置 (根据实际情况修改)
fs = 10e6;              % 采样率 10 MHz
hop_rate = 1000;        % 跳频速率 (每秒跳多少次)
samples_per_hop = fs / hop_rate; % 每跳的样点数
f_hop_list = [1e6, -2e6, 3e6, 0.5e6, -1.5e6]; % 跳频图案 (f0, f1, ... fn)
N_hops = length(f_hop_list); % 跳频总数

% 模拟信号参数 (基带信号带宽)
bw_signal = 200e3; 

%% 2. 模拟生成跳频IQ数据 (如果你已有文件，可跳过此步)
disp('正在生成模拟跳频信号...');
total_samples = N_hops * samples_per_hop;
t_total = (0:total_samples-1).' / fs;
raw_iq_data = zeros(total_samples, 1);

% 生成基带数据 (例如 QPSK 或 简单单音)
% 这里用带限噪声模拟基带信号
baseband_sig = (randn(total_samples, 1) + 1j*randn(total_samples, 1));
[b_low, a_low] = butter(6, bw_signal/(fs/2));
baseband_sig = filter(b_low, a_low, baseband_sig);

% 调制到跳频频率上
for i = 1:N_hops
    idx_start = (i-1)*samples_per_hop + 1;
    idx_end = i*samples_per_hop;
    
    % 当前跳的频率
    f_c = f_hop_list(i);
    
    % 时间片段 (必须使用全局时间以保持相位连续性，或者根据发射端机制决定)
    t_seg = t_total(idx_start:idx_end);
    
    % 搬移频率: Baseband * exp(j*2*pi*fc*t)
    raw_iq_data(idx_start:idx_end) = baseband_sig(idx_start:idx_end) .* exp(1j * 2 * pi * f_c * t_seg);
end

% 模拟保存为文件 (模拟你的离线文件)
save('hopping_iq_data.mat', 'raw_iq_data');
clear raw_iq_data; % 清除内存，模拟完全离线读取

%% 3. 核心代码：离线数据读取与解跳拼接
disp('开始处理离线数据...');

% --- 步骤 A: 加载数据 ---
% 实际中可能是 fread 打开二进制 dat 文件
load('hopping_iq_data.mat'); % 加载变量 raw_iq_data
raw_data = raw_iq_data; % 假设加载后的变量名为 raw_iq_data

% 预分配输出内存
dehopped_iq = zeros(size(raw_data));

% --- 步骤 B: 循环解跳 ---
% 假设我们已知：采样率 fs, 每跳点数 samples_per_hop, 频率序列 f_hop_list

for k = 1:N_hops
    % 1. 确定当前跳的数据索引范围
    idx_start = (k-1)*samples_per_hop + 1;
    idx_end = min(k*samples_per_hop, length(raw_data)); % 防止越界
    
    if idx_start > length(raw_data)
        break;
    end
    
    % 2. 提取当前跳的原始IQ数据
    segment_iq = raw_data(idx_start:idx_end);
    
    % 3. 生成反向本振 (NCO)
    % 关键点：为了保证解跳后的相位在时间轴上是物理对应的，
    % 这里的 t 应该对应原始信号的"绝对时间"或"相对文件开始的时间"。
    current_freq = f_hop_list(k);
    t_axis = (idx_start-1 : idx_end-1).' / fs; 
    
    % 下变频向量: exp(-j * 2 * pi * f_k * t)
    nco = exp(-1j * 2 * pi * current_freq * t_axis);
    
    % 4. 混频 (解跳) 并 拼接
    dehopped_iq(idx_start:idx_end) = segment_iq .* nco;
end

%% 4. 后处理 (可选：低通滤波)
% 解跳后，通常需要滤除可能存在的残留高频分量或噪声
% 如果是纯数学搬移且无干扰，此步可忽略，但在实际信号中很必要
filter_bw = bw_signal * 1.2; % 滤波器带宽略大于信号带宽
[b_lpf, a_lpf] = butter(8, filter_bw/(fs/2));
clean_iq = filter(b_lpf, a_lpf, dehopped_iq);

%% 5. 结果可视化验证
figure('Position', [100, 100, 1000, 600]);

subplot(2,1,1);
spectrogram(raw_data, 256, 128, 256, fs, 'yaxis', 'centered');
title('原始跳频信号时频图');
xlabel('时间 (s)'); ylabel('频率 (Hz)');

subplot(2,1,2);
spectrogram(dehopped_iq, 256, 128, 256, fs, 'yaxis', 'centered');
title('解跳拼接后信号时频图 (全零频)');
xlabel('时间 (s)'); ylabel('频率 (Hz)');

disp('处理完成。');