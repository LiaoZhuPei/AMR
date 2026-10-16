% =========================================================================
%                restore_universal_final.m
%       (最终的、真正通用的、两步式信号还原接收机)
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

start_sample = 20001; % 您可以修改这个值来探索不同的片段
y_distorted_signal = complex_signal_full(start_sample : start_sample + segment_length - 1);


%% ======================= 最终的、两步式还原过程 =======================

fprintf('--- 开始两步式通用还原 ---\n');

% --- 步骤 A: 宏观校正 (包络移除) ---
% 这一步负责处理CFO、增益和大部分DC偏置

filter_window_size = 51; 
estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');
epsilon = 1e-9;
y_macro_corrected = y_distorted_signal ./ (estimated_envelope + epsilon);

% --- 步骤 B: 精细校正 (精确的零均值化) ---
% 这一步确保I和Q通道都关于X轴对称

% 1. 计算宏观校正后信号的残余直流偏置
residual_dc_offset = mean(y_macro_corrected);

% 2. 减去这个残余偏置
y_final_restored = y_macro_corrected - residual_dc_offset;

% --- 步骤 C: (可选) 最终归一化 ---
% 将I通道的“ON”状态的平均幅度缩放到1
i_channel = real(y_final_restored);
on_state_points = i_channel(i_channel > mean(i_channel)); % 简单地取高于均值的点
if ~isempty(on_state_points)
    norm_factor = mean(on_state_points);
    y_final_restored = y_final_restored / norm_factor;
end


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

% 原始失真信号
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title('1. 原始信号片段');
grid on; legend('I', 'Q');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

% 最终还原信号
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 最终还原的近似理想信号 (两步法)');
grid on; legend('I', 'Q'); ylim([-2 2]);

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('还原后信号星座图 (两步法)');
grid on; axis equal; xlim([-0.5 1.5]); ylim([-1 1]);