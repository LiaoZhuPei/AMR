% =========================================================================
%                experiment_advanced_rerotation.m
%       (一个先按您要求还原信号，再施加随机相位以供对比的实验脚本)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 加载您的真实 .wav 文件 ---
% (这部分代码与您提供的完全相同)
wav_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'; 
segment_length = 1024;
[y_raw, ~] = audioread(wav_file_path);
complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);
start_sample = randi([1, length(complex_signal_full) - segment_length]);
y_distorted_signal = complex_signal_full(start_sample : start_sample + segment_length - 1);

%% ======================= 步骤 2: 对原始信号进行还原 =======================
% (这部分代码也与您提供的完全相同)

fprintf('--- 开始进行最终的分通道重定位还原 ---\n');

% --- 步骤 A: 宏观校正 (复数包络移除) ---
filter_window_size = 301; 
estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');
epsilon = 1e-9;
y_envelope_removed = y_distorted_signal ./ (estimated_envelope + epsilon);
y_centered = y_envelope_removed - mean(y_envelope_removed);

% --- 步骤 B: 分通道精细重定位 ---
i_channel_processed = real(y_centered);
q_channel_processed = imag(y_centered);

min_val_i = min(i_channel_processed);
i_channel_final = i_channel_processed - min_val_i;

q_channel_final = q_channel_processed;

% 组合成还原后的信号
y_restored = i_channel_final + 1i * q_channel_final;

%% ======================= 步骤 3: 在还原后的信号上施加新的随机相位 =======================

% 生成一个 [0, 2*pi] 范围内的随机相位角
random_phase_rad = rand() * 2 * pi;
fprintf('--- 正在对还原后的信号施加一个新的随机相位: %.2f 度 ---\n', rad2deg(random_phase_rad));

% 创建旋转因子
random_rotation_factor = exp(1i * random_phase_rad);

% 将旋转因子乘到已还原的信号上
y_restored_and_rerotated = y_restored .* random_rotation_factor;


%% ======================= 步骤 4: 可视化对比 (三组图) =======================
figure('Position', [50, 50, 1800, 800]); % 调整图形窗口大小以容纳三组图

% --- 第一组: 原始失真信号 ---
subplot(2, 3, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title('1a. 原始失真信号'); grid on; legend('I', 'Q');

subplot(2, 3, 4);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('1b. 原始信号星座图'); grid on; axis equal;

% --- 第二组: 还原后的信号 (您的标准形态) ---
subplot(2, 3, 2);
plot(real(y_restored), 'b'); hold on; plot(imag(y_restored), 'r');
title('2a. 还原后的信号 (I重定位, Q保留)'); grid on; legend('I', 'Q'); ylim([-1.5 2.5]);

subplot(2, 3, 5);
scatter(real(y_restored), imag(y_restored), 10, 'b', 'filled');
title('2b. 还原后信号星座图'); grid on; axis equal; xlim([-0.5 2.5]);

% --- 第三组: 还原后 + 再旋转的信号 ---
subplot(2, 3, 3);
plot(real(y_restored_and_rerotated), 'b'); hold on; plot(imag(y_restored_and_rerotated), 'r');
title(['3a. 还原后再旋转 (' num2str(round(rad2deg(random_phase_rad))) '°) 的信号']); 
grid on; legend('I', 'Q');

subplot(2, 3, 6);
scatter(real(y_restored_and_rerotated), imag(y_restored_and_rerotated), 10, 'b', 'filled');
title('3b. 还原后再旋转的星座图'); grid on; axis equal;