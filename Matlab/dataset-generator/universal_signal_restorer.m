% =========================================================================
%                universal_signal_restorer.m
%       (一个可以处理.wav或.mat文件，并提供多种还原算法的通用工具)
% =========================================================================
clc, clear, close all

%% --- 1. 用户配置区 ---

% --- 输入文件配置 ---
% 您可以在这里输入 .wav 或 .mat 文件的路径
%input_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav';
input_file_path = 'output_mat_files_ASK2_matlab/segment_000001.mat'; % <<--- 修改这里！

% --- 算法选择 ---
% 'two_step': 我们最终的“包络移除 + 零均值”两步法。通用性好，推荐。
% 'pca': 基于主成分分析的“平移+旋转”法。能更好地保持波形形状。
algorithm_choice = 'two_step'; % <<--- 在这里选择 'two_step' 或 'pca'

% --- WAV文件处理参数 (仅当输入为.wav时有效) ---
segment_length = 1024;
start_sample = 'random'; % 'random' 或一个具体的数字，如 20001


%% --- 2. 自动文件加载 ---

fprintf('正在处理文件: %s\n', input_file_path);
[~, ~, ext] = fileparts(input_file_path); % 获取文件扩展名

if strcmpi(ext, '.wav')
    % --- WAV 文件加载逻辑 ---
    try
        [y_raw, fs] = audioread(input_file_path);
    catch ME
        error('无法读取WAV文件。错误: %s', ME.message);
    end
    if size(y_raw, 2) ~= 2, error('WAV文件必须是立体声。'); end
    complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);
    
    if strcmpi(start_sample, 'random')
        max_start = length(complex_signal_full) - segment_length + 1;
        if max_start < 1, error('信号总长度小于片段长度。'); end
        start_idx = randi([1, max_start]);
    else
        start_idx = start_sample;
    end
    fprintf('从WAV文件中截取片段，起始点: %d\n', start_idx);
    y_distorted_signal = complex_signal_full(start_idx : start_idx + segment_length - 1);
    
elseif strcmpi(ext, '.mat')
    % --- MAT 文件加载逻辑 ---
    try
        mat_contents = load(input_file_path);
    catch ME
        error('无法读取MAT文件。错误: %s', ME.message);
    end
    if isfield(mat_contents, 'data_save')
        iq_matrix = mat_contents.data_save;
    else
        error('在.mat文件中未找到名为 "data_save" 的变量。');
    end
    if size(iq_matrix, 1) == 2
        y_distorted_signal = iq_matrix(1, :) + 1i * iq_matrix(2, :);
    else
        error('不支持的.mat文件内数据格式。');
    end
else
    error('不支持的文件类型: %s。请输入.wav或.mat文件。', ext);
end

L = length(y_distorted_signal);

%% --- 3. 根据选择，执行相应的还原算法 ---

fprintf('选择的还原算法: %s\n', algorithm_choice);

if strcmpi(algorithm_choice, 'two_step')
    % --- “两步法”还原 ---
    filter_window_size = 51; 
    estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');
    epsilon = 1e-9;
    y_macro_corrected = y_distorted_signal ./ (estimated_envelope + epsilon);
    residual_dc_offset = mean(y_macro_corrected);
    y_final_restored = y_macro_corrected - residual_dc_offset;
    % 可选归一化
    i_channel = real(y_final_restored);
    on_state_points = i_channel(i_channel > mean(i_channel));
    if ~isempty(on_state_points)
        norm_factor = mean(on_state_points);
        if abs(norm_factor) > 1e-6, y_final_restored = y_final_restored / norm_factor; end
    end
    
elseif strcmpi(algorithm_choice, 'pca')
    % --- “PCA法”还原 ---
    dc_offset_estimated = mean(y_distorted_signal);
    y_dc_corrected = y_distorted_signal - dc_offset_estimated;
    data_for_pca = [real(y_dc_corrected)', imag(y_dc_corrected)'];
    [coeff, ~] = pca(data_for_pca);
    principal_vector = coeff(:, 1);
    phase_offset_rad = atan2(principal_vector(2), principal_vector(1));
    phase_correction_factor = exp(-1i * phase_offset_rad);
    y_final_restored = y_dc_corrected .* phase_correction_factor;
    % 可选翻转和归一化
    if mean(real(y_final_restored)) < 0, y_final_restored = y_final_restored * -1; end
    y_final_restored = y_final_restored / max(abs(real(y_final_restored)));
    
else
    error('未知的算法选择: %s。请选择 "two_step" 或 "pca"。', algorithm_choice);
end

%% --- 4. 可视化对比 ---
figure('Position', [50, 50, 1600, 800]);

subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title(['1. 原始信号片段 (来自 ' strrep(input_file_path, '_', '\_') ')']);
grid on; legend('I', 'Q');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title(['2. 最终还原的近似理想信号 (' algorithm_choice ')']);
grid on; legend('I', 'Q'); ylim([-2 2]);

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title(['还原后信号星座图 (' algorithm_choice ')']);
grid on; axis equal; xlim([-0.5 1.5]); ylim([-1 1]);