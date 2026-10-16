% =========================================================================
%                process_real_mat_file.m
%       (用于处理您.mat文件的信号还原脚本)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 加载您的 .mat 文件 ---

% 【用户配置区】
mat_file_path = 'E:/M_Project/P/dataset-generator/output_mat_files_ASK2_matlab/segment_000004.mat'; % <<--- 请将这里替换为您.mat文件的真实路径！

% 加载.mat文件。
% !!! 注意：请确保您知道.mat文件中存储IQ数据的变量名。
% 常见的变量名有 'data_save', 'iq_data', 'y' 等。
% 如果下面的 'iq_data' 不对，请修改为正确的变量名。
try
    mat_contents = load(mat_file_path);
    % 假设变量名为 'iq_data'，并且格式为 2x1024
    if isfield(mat_contents, 'iq_data')
        iq_matrix = mat_contents.iq_data;
    % 假设变量名为 'data_save'，并且格式为 1x2x1024
    elseif isfield(mat_contents, 'data_save')
        iq_squeezed = squeeze(mat_contents.data_save);
        iq_matrix = iq_squeezed; % 应该已经是 2x1024
    else
        % 如果变量名不确定，可以尝试获取第一个字段
        field_names = fieldnames(mat_contents);
        iq_matrix = mat_contents.(field_names{1});
        fprintf('警告: 未找到指定变量名, 已使用第一个变量 "%s"。\n', field_names{1});
    end

    fprintf('成功读取文件: %s\n', mat_file_path);
catch ME
    error('无法读取MAT文件。请检查路径和变量名。错误信息: %s', ME.message);
end

% 检查并转换数据格式为复数信号
if size(iq_matrix, 1) == 2 % 格式为 2xN
    y_distorted_signal = iq_matrix(1, :) + 1i * iq_matrix(2, :);
elseif size(iq_matrix, 2) == 2 % 格式为 Nx2
    y_distorted_signal = iq_matrix(:, 1) + 1i * iq_matrix(:, 2);
else
    error('不支持的.mat文件内数据格式。需要是 2xN 或 Nx2。');
end
% 确保是行向量
y_distorted_signal = y_distorted_signal(:).'; 
L = length(y_distorted_signal);

%% ======================= 包络移除过程 =======================

fprintf('--- 开始移除信号包络/直流偏置 ---\n');

% 【用户可调参数】: 滤波器窗口大小
filter_window_size = 51; 

% --- 步骤 A: 估计包络/直流偏置 ---
estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');

% --- 步骤 B: 移除包络/直流偏置 ---
epsilon = 1e-9;
y_envelope_removed = y_distorted_signal ./ (estimated_envelope + epsilon);

% --- 步骤 C: 幅度归一化 ---
y_final_restored = y_envelope_removed / mean(abs(y_envelope_removed));


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

% 原始失真信号
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; 
plot(imag(y_distorted_signal), 'r');
plot(real(estimated_envelope), 'k--', 'LineWidth', 2); 
plot(imag(estimated_envelope), 'k--', 'LineWidth', 2);
title('1. 原始.mat文件信号片段 与 估计出的包络(直流偏置)');
grid on; legend('I', 'Q', '包络估计');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

% 最终还原信号
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 移除了包络/直流偏置后的信号');
grid on; legend('I', 'Q'); ylim([-2 2]);

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('移除包络/直流偏置后信号星座图');
grid on; axis equal; xlim([-2 2]); ylim([-2 2]);