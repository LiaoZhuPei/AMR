% =========================================================================
%                signal_diagnostics.m (最终稳健版)
% =========================================================================
clc, clear, close all

%% --- 1. 用户配置区 ---
input_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'; % <<--- 修改这里！
segment_length = 1024;
start_sample = 'random'; 

%% --- 2. 自动文件加载 ---
% (这部分代码与之前版本完全相同)
fprintf('正在加载和分析文件: %s\n', input_file_path);
[~, ~, ext] = fileparts(input_file_path);
if strcmpi(ext, '.wav')
    try [y_raw, ~] = audioread(input_file_path); catch ME, error('无法读取WAV文件。错误: %s', ME.message); end
    if size(y_raw, 2) ~= 2, error('WAV文件必须是立体声。'); end
    complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);
    if strcmpi(start_sample, 'random')
        max_start = length(complex_signal_full) - segment_length + 1;
        if max_start < 1, error('信号总长度小于片段长度。'); end
        start_idx = randi([1, max_start]);
    else
        start_idx = start_sample;
    end
    y_signal = complex_signal_full(start_idx : start_idx + segment_length - 1);
elseif strcmpi(ext, '.mat')
    try mat_contents = load(input_file_path); catch ME, error('无法读取MAT文件。错误: %s', ME.message); end
    if isfield(mat_contents, 'data_save'), iq_matrix = mat_contents.data_save; else error('在.mat文件中未找到名为 "data_save" 的变量。'); end
    if size(iq_matrix, 1) == 2, y_signal = iq_matrix(1, :) + 1i * iq_matrix(2, :); else error('不支持的.mat文件内数据格式。'); end
else
    error('不支持的文件类型: %s。', ext);
end

%% ======================= 3. 智能诊断过程 (稳健版) =======================

fprintf('\n--- 开始进行信号诊断 ---\n');

% --- 步骤 A: 计算“偏移度” ---
dc_offset_vector = mean(y_signal);
offset_magnitude = abs(dc_offset_vector);
rms_val = sqrt(mean(abs(y_signal).^2));
normalized_offset = offset_magnitude / rms_val;
fprintf('归一化直流偏移幅度: %.4f\n', normalized_offset);

% --- 步骤 B: 计算“弯曲度” (线性度) ---
y_centered = y_signal - dc_offset_vector;
data_for_pca = [real(y_centered)', imag(y_centered)'];
[coeff, ~, ~, ~, explained] = pca(data_for_pca);

% 【关键修正】：在使用'explained'之前，检查它是否为空以及其大小
if isempty(explained)
    % 如果 explained 是空的，说明数据几乎是完美的1D，弯曲度为0
    linearity_metric = 0;
    fprintf('PCA分析: 数据高度共线，explained变量为空。\n');
elseif length(explained) < 2
    % 如果 explained 只有一个元素，说明第二主成分能量为0
    linearity_metric = 0;
    fprintf('PCA分析: 数据几乎是一维的，第二主成分能量为0。\n');
else
    % 只有当 explained 至少有两个元素时，我们才安全地访问它
    energy_pc2_percent = explained(2);
    linearity_metric = energy_pc2_percent;
end
fprintf('星座图弯曲度 (次要方向能量占比): %.2f%%\n', linearity_metric);

%% --- 4. 做出诊断决策 ---
% (这部分代码与之前版本完全相同)
fprintf('\n--- 诊断报告 ---\n');
offset_threshold = 0.2;
linearity_threshold = 5.0; 
is_dc_dominant = false;
is_cfo_dominant = false;

if normalized_offset > offset_threshold
    fprintf('[+] 检测到显著的直流偏置。\n');
    is_dc_dominant = true;
else
    fprintf('[-] 未检测到显著的直流偏置。\n');
end

if linearity_metric > linearity_threshold
    fprintf('[+] 检测到显著的星座图弯曲 (表明存在CFO或频率漂移)。\n');
    is_cfo_dominant = true;
else
    fprintf('[-] 星座图接近直线 (表明无显著CFO)。\n');
end

fprintf('\n最终诊断: ');
if is_cfo_dominant
    fprintf('信号类型为【载波偏移主导】。\n');
else
    fprintf('信号类型为【直流偏置主导】。\n');
end

%% --- 5. 可视化 ---
% (这部分代码与之前版本完全相同)
figure('Position', [100, 100, 1000, 500]);
scatter(real(y_signal), imag(y_signal), 10, 'b', 'filled');
title(['原始信号星座图 (诊断: ' get_diagnosis_string(is_cfo_dominant) ')']);
grid on; axis equal;
xlabel('I'); ylabel('Q');

function str = get_diagnosis_string(is_cfo)
    if is_cfo
        str = 'CFO主导';
    else
        str = 'DC偏置主导';
    end
end