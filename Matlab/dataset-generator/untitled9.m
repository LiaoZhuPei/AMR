% =========================================================================
%                universal_restorer_final.m
%       (一个基于二次相位回归的、真正通用的信号还原器)
% =========================================================================
clc, clear, close all

%% --- 1. 用户配置区 ---
% 您可以输入任何一个 .wav 或 .mat 文件的路径
% input_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'; % 尝试CFO主导
input_file_path = 'output_mat_files_ASK2_matlab/segment_000001.mat'; % 尝试静态相位主导

%% --- 2. 自动文件加载 ---
% (与之前的通用脚本相同)
% ... (省略，与之前代码块完全相同) ...
fprintf('正在加载和分析文件: %s\n', input_file_path);
[~, ~, ext] = fileparts(input_file_path);
if strcmpi(ext, '.wav')
    segment_length = 1024;
    try [y_raw, ~] = audioread(input_file_path); catch ME, error('无法读取WAV文件。错误: %s', ME.message); end
    if size(y_raw, 2) ~= 2, error('WAV文件必须是立体声。'); end
    complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);
    start_idx = randi([1, length(complex_signal_full) - segment_length + 1]);
    y_distorted_signal = complex_signal_full(start_idx : start_idx + segment_length - 1);
elseif strcmpi(ext, '.mat')
    try mat_contents = load(input_file_path); catch ME, error('无法读取MAT文件。错误: %s', ME.message); end
    if isfield(mat_contents, 'data_save'), iq_matrix = mat_contents.data_save; else error('在.mat文件中未找到名为 "data_save" 的变量。'); end
    if size(iq_matrix, 1) == 2, y_distorted_signal = iq_matrix(1, :) + 1i * iq_matrix(2, :); else error('不支持的.mat文件内数据格式。'); end
else
    error('不支持的文件类型: %s。', ext);
end
L = length(y_distorted_signal);


%% ======================= 终极的、通用的还原过程 =======================

fprintf('--- 开始进行通用的、基于二次相位回归的还原 ---\n');

% --- 步骤 A: 预处理 (总是执行) ---
y_proc = conj(y_distorted_signal); % 盲猜存在频谱反转
y_proc = y_proc / sqrt(mean(abs(y_proc).^2)); % 归一化
y_proc = y_proc - mean(y_proc); % 中心化

% --- 步骤 B: 通用的二次相位回归与校正 ---
phase_all = angle(y_proc);
unwrapped_phase_all = unwrap(phase_all);
amp_threshold = (max(abs(y_proc)) + min(abs(y_proc))) * 0.5;
reliable_indices = find(abs(y_proc) > amp_threshold);

y_final_restored = y_proc; % 默认值，以防校正失败

if length(reliable_indices) > 50
    n_reliable = reliable_indices' - 1; 
    phase_for_regression = unwrapped_phase_all(reliable_indices);
    
    % 拟合二次曲线。这个步骤是通用的！
    coeffs = polyfit(n_reliable, phase_for_regression, 2);
    p2 = coeffs(1); p1 = coeffs(2);
    fprintf('估计的相位模型: %.2e*n^2 + %.4f*n + ...\n', p2, p1);
    
    % 创建并应用校正因子
    n_vec = 0:(L-1);
    phase_correction_curve = p2*n_vec.^2 + p1*n_vec;
    correction_factor = exp(-1i * phase_correction_curve);
    y_cfo_corrected = y_proc .* correction_factor;
    
    % 最终相位对齐
    reliable_indices_final = find(abs(y_cfo_corrected) > amp_threshold);
    if ~isempty(reliable_indices_final)
        on_state_points = y_cfo_corrected(reliable_indices_final);
        phase_offset_estimated = mean(angle(on_state_points));
        phase_correction_factor = exp(-1i * phase_offset_estimated);
        y_final_restored = y_cfo_corrected .* phase_correction_factor;
    end
end


%% ======================= 可视化对比 =======================
% (与之前版本完全相同)
figure('Position', [50, 50, 1600, 800]);
% ... (省略，与之前代码块完全相同) ...
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title(['1. 原始信号片段']); grid on; legend('I', 'Q');
subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图'); grid on; axis equal;
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 最终通用还原后的信号'); grid on; ylim([-1.5 1.5]);
subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('还原后信号星座图'); grid on; axis equal; xlim([-0.5 1.5]);