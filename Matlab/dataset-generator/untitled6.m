% =========================================================================
%                restore_pure_translation.m
%       (一个只进行平移，不改变任何形状和幅值的最终脚本)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 加载您的 .mat 文件 ---
mat_file_path = 'output_mat_files_ASK2_matlab/segment_000001.mat'; % <<--- 修改这里！
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

%% ======================= 纯粹的平移操作 =======================

fprintf('--- 开始进行纯粹的平移操作 ---\n');

% 1. 分离原始的I和Q通道
i_channel_original = real(y_distorted_signal);
q_channel_original = imag(y_distorted_signal);

% --- 【I通道处理】：将其移动到以y=0为起始位置 ---
% 计算I通道的最小值
min_i = min(i_channel_original);
% 计算需要施加的平移量
translation_i = -min_i;
% 应用平移
i_channel_final = i_channel_original + translation_i;

% --- 【Q通道处理】：将其移动到关于x轴对称的位置 ---
% 一个关于x轴对称的信号，其数学平均值必须为零。
% 计算Q通道的平均值
mean_q = mean(q_channel_original);
% 计算需要施加的平移量
translation_q = -mean_q;
% 应用平移
q_channel_final = q_channel_original + translation_q;

% 组合成最终的信号
y_final_restored = i_channel_final + 1i * q_channel_final;

fprintf('I通道已平移 %.2f，使其最小值为0。\n', translation_i);
fprintf('Q通道已平移 %.2f，使其均值为0。\n', translation_q);

%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

% 原始失真信号
subplot(2, 2, 1);
plot(i_channel_original, 'b'); hold on; plot(q_channel_original, 'r');
title(['1. 原始 .mat 文件信号片段']);
grid on; legend('I', 'Q');

% 最终校正信号
subplot(2, 2, 2);
plot(i_channel_final, 'b'); hold on; plot(q_channel_final, 'r');
title('2. 仅位置平移后的信号');
grid on; legend('I', 'Q');