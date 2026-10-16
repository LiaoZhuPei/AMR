% =========================================================================
%                restore_envelope_then_translate.m
%       (一个先移除包络，再进行精细平移的最终脚本)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 加载您的真实 .wav 文件 ---
wav_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'; 
segment_length = 1024;
[y_raw, ~] = audioread(wav_file_path);
complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);
start_sample = randi([1, length(complex_signal_full) - segment_length]);
y_distorted_signal = complex_signal_full(start_sample : start_sample + segment_length - 1);

%% ======================= 最终的两阶段还原过程 =======================

fprintf('--- 开始进行最终的两阶段还原 ---\n');

% --- 阶段一: 宏观校正 (复数包络移除) ---
fprintf('阶段一：正在移除复数包络...\n');
filter_window_size = 51; 
estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');
epsilon = 1e-9;
y_envelope_removed = y_distorted_signal ./ (estimated_envelope + epsilon);

% --- 阶段二: 精细重定位 (独立I/Q平移) ---
fprintf('阶段二：正在进行独立的I/Q通道精细平移...\n');

% 1. 提取移除了宏观包络后的I和Q通道
i_channel_processed = real(y_envelope_removed);
q_channel_processed = imag(y_envelope_removed);

% 2. 【I通道处理】: 将其移动到以y=0为起始位置
min_i = min(i_channel_processed);
translation_i = -min_i;
i_channel_final = i_channel_processed + translation_i;

% 3. 【Q通道处理】: 将其移动到关于x轴对称的位置
mean_q = mean(q_channel_processed);
translation_q = -mean_q;
q_channel_final = q_channel_processed + translation_q;

% 4. 组合成最终的信号
y_final_restored = i_channel_final + 1i * q_channel_final;

% 【关键】：全程没有进行任何归一化或缩放操作。

%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; 
plot(imag(y_distorted_signal), 'r');
plot(real(estimated_envelope), 'k--', 'LineWidth', 2);
title(['1. 原始信号片段与包络 (起始点: ' num2str(start_sample) ')']);
grid on; legend('I', 'Q', '包络估计');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; 
plot(imag(y_final_restored), 'r');
title('2. 最终还原的信号 (移除包络+精细平移)');
grid on; legend('I', 'Q');

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('还原后信号星座图');
grid on; axis equal;