% =========================================================================
%                remove_envelope_simple.m
%       (随机处理.wav文件中不同片段的实用脚本)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 加载您的真实 .wav 文件 ---

% 【用户配置区】
wav_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'; % <<--- 修改这里！
segment_length = 1024; % 定义要处理的片段长度

% 使用 audioread 函数读取 .wav 文件
try
    [y_raw, fs] = audioread(wav_file_path);
    fprintf('成功读取文件: %s\n', wav_file_path);
catch ME
    error('无法读取WAV文件。请检查路径是否正确。错误信息: %s', ME.message);
end

if size(y_raw, 2) ~= 2, error('WAV文件必须是立体声。'); end

% 将I和Q通道组合成复数信号
complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);
total_samples = length(complex_signal_full);
fprintf('文件总长度为: %d 个采样点。\n', total_samples);

% --- 【关键修改】：从文件中随机选择一个片段 ---
% 计算允许的最大起始点，以确保能截取一个完整的片段
max_start_sample = total_samples - segment_length + 1;
if max_start_sample < 1
    error('信号总长度小于一个片段的长度，无法处理。');
end

% 在允许的范围内，随机生成一个起始点
start_sample = randi([1, max_start_sample]);
fprintf('本次将处理从采样点 %d 开始的片段。\n', start_sample);

% 从长信号中截取随机选择的片段
y_distorted_signal = complex_signal_full(start_sample : start_sample + segment_length - 1);


%% ======================= 包络移除过程 =======================

fprintf('--- 开始移除信号包络 ---\n');

% 滤波器窗口大小
filter_window_size = 51; 

% 【关键修改 1】：新增正则化因子
% 这个小常数可以防止分母过小，从而抑制幅度放大效应。
% 它的值通常是根据信号的平均功率来确定的。
regularization_factor = 0.1 * mean(abs(y_distorted_signal)); % 例如，设为平均幅度的10%

% --- 步骤 A: 估计包络 ---
estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');

% --- 步骤 B: 移除包络 ---
% 【关键修改 2】：在分母上加上正则化因子
y_envelope_removed = y_distorted_signal ./ (estimated_envelope + regularization_factor);

% --- 步骤 C: (可选) 幅度归一化 ---
y_final_restored = y_envelope_removed / mean(abs(y_envelope_removed));


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

% 原始失真信号
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; 
plot(imag(y_distorted_signal), 'r');
plot(real(estimated_envelope), 'k--', 'LineWidth', 2); 
plot(imag(estimated_envelope), 'k--', 'LineWidth', 2);
title(['1. 原始信号片段 (起始点: ' num2str(start_sample) ')']);
grid on; legend('I', 'Q', '包络估计');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

% 最终还原信号
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 移除了包络后的信号');
grid on; legend('I', 'Q'); ylim([-2 2]);

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('移除包络后信号星座图');
grid on; axis equal; xlim([-2 2]); ylim([-2 2]);