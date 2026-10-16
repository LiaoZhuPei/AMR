% =========================================================================
%                remove_envelope_simple.m
%       (使用低通滤波来移除信号包络的实用脚本)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 加载您的真实 .wav 文件 ---
wav_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'; % <<--- 修改这里！
segment_length = 1024;
try
    [y_raw, fs] = audioread(wav_file_path);
    fprintf('成功读取文件: %s\n', wav_file_path);
catch ME
    error('无法读取WAV文件。错误信息: %s', ME.message);
end
if size(y_raw, 2) ~= 2, error('WAV文件必须是立体声。'); end
complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);

start_sample = 20001;
if (start_sample + segment_length - 1) > length(complex_signal_full)
    error('信号长度不足，无法截取指定片段。');
end
y_distorted_signal = complex_signal_full(start_sample : start_sample + segment_length - 1);


%% ======================= 包络移除过程 =======================

fprintf('--- 开始移除信号包络 ---\n');

% 【用户可调参数】: 滤波器窗口大小
% 这个参数决定了平滑的程度。
% 窗口越大，估计出的包络越平滑，但也可能丢失一些细节。
% 窗口大小应该大于一个ASK符号的长度 (sps)，但远小于整个片段的长度。
% 51 是一个不错的起始猜测值。
filter_window_size = 51; % <<--- 调整这个值来改变平滑效果

% --- 步骤 A: 估计包络 ---
% 使用 'smooth' 函数进行移动平均滤波来估计慢变的包络。
% 我们对复数信号直接进行滤波。
estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');

% --- 步骤 B: 移除包络 ---
% 用原始信号除以估计出的包络。
% 为了防止除以零，我们在分母上加一个很小的数 (epsilon)。
epsilon = 1e-9;
y_envelope_removed = y_distorted_signal ./ (estimated_envelope + epsilon);

% --- 步骤 C: (可选) 幅度归一化 ---
% 移除包络后，信号的幅度会集中在1附近。我们可以将其归一化。
y_final_restored = y_envelope_removed / mean(abs(y_envelope_removed));


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

% 原始失真信号
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; 
plot(imag(y_distorted_signal), 'r');
plot(real(estimated_envelope), 'k--', 'LineWidth', 2); % 叠加绘制估计出的包络
plot(imag(estimated_envelope), 'k--', 'LineWidth', 2);
title('1. 原始信号 与 估计出的包络');
grid on; legend('I', 'Q', '包络估计');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

% 最终还原信号
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 移除了包络后的信号');
grid on; legend('I', 'Q'); ylim([-2 2]); % 固定Y轴

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('移除包络后信号星座图');
grid on; axis equal; xlim([-2 2]); ylim([-2 2]);