% =========================================================================
%                process_mat_pca_based.m
%       (基于PCA的、保持波形形状的智能信号还原脚本)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 加载您的 .mat 文件 ---
mat_file_path = 'output_mat_files_ASK2_matlab/segment_000005.mat'; % <<--- 修改这里！
try
    mat_contents = load(mat_file_path);
    fprintf('成功读取文件: %s\n', mat_file_path);
catch ME
    error('无法读取MAT文件。错误信息: %s', ME.message);
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
L = length(y_distorted_signal);


%% ======================= 基于PCA的智能还原法 =======================

fprintf('--- 开始基于PCA的智能还原 ---\n');

% --- 步骤 A: 中心化 (平移) - 校正直流偏置 ---
dc_offset_estimated = mean(y_distorted_signal);
y_dc_corrected = y_distorted_signal - dc_offset_estimated;

% --- 步骤 B: 对齐 (旋转) - 使用PCA找到主方向并校正相位 ---
% 准备数据进行PCA：需要一个 Nx2 的实数矩阵
data_for_pca = [real(y_dc_corrected)', imag(y_dc_corrected)'];

% 使用 pca 函数计算主成分。
% 'coeff' 是一个2x2的矩阵，它的第一列就是第一主成分的向量！
[coeff, ~, ~, ~, ~, mu] = pca(data_for_pca);

% 提取第一主成分向量
principal_vector = coeff(:, 1);

% 计算这个向量的角度。这个角度就是我们信号的固定相位偏移。
% atand 提供以度为单位的角度, atan2更稳健
phase_offset_rad = atan2(principal_vector(2), principal_vector(1));
fprintf('通过PCA估计出的固定相位偏移为: %.2f 度\n', rad2deg(phase_offset_rad));

% 创建一个反向旋转因子
phase_correction_factor = exp(-1i * phase_offset_rad);

% 将反向旋转应用到已中心化的信号上
y_final_restored = y_dc_corrected .* phase_correction_factor;

% --- 步骤 C: (可选) 最终归一化和符号翻转检查 ---
% 旋转后，大部分能量都在I轴上
i_channel = real(y_final_restored);
% 检查信号是否主要在负半轴，如果是，则翻转180度
if mean(i_channel) < 0
    y_final_restored = y_final_restored * -1;
end
% 归一化，使最大幅度为1
y_final_restored = y_final_restored / max(abs(real(y_final_restored)));


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title('1. 原始 .mat 文件信号片段');
grid on; legend('I', 'Q');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 最终还原的近似理想信号 (PCA法)');
grid on; legend('I', 'Q'); ylim([-1.5 1.5]);

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('还原后信号星座图 (PCA法)');
grid on; axis equal; xlim([-0.2 1.2]); ylim([-1 1]);