% =========================================================================
%                batch_restore_wav_to_csv.m (最终稳健版)
% =========================================================================
clc, clear, close all

%% --- 1. 用户配置区 ---
wav_file_path = 'SDRconnect_IQ_20250906_204407_300000000HZ.wav'; 
segment_length = 1024;
num_files_to_generate = 200;
output_folder = 'restored_wav_signals_csv';

%% --- 2. 准备工作 ---
if ~exist(output_folder, 'dir'), mkdir(output_folder); end
try
    [y_raw, ~] = audioread(wav_file_path);
    fprintf('成功读取文件: %s\n', wav_file_path);
catch ME
    error('无法读取WAV文件。错误: %s', ME.message);
end
if size(y_raw, 2) ~= 2, error('WAV文件必须是立体声。'); end
complex_signal_full = y_raw(:, 1) + 1i * y_raw(:, 2);
total_samples = length(complex_signal_full);
max_start_sample = total_samples - segment_length + 1;
if max_start_sample < 1, error('信号总长度小于片段长度。'); end

%% --- 3. 主循环，批量处理和保存 ---
fprintf('开始批量生成和还原 %d 个信号片段...\n', num_files_to_generate);

for i = 1:num_files_to_generate
    
    start_sample = randi([1, max_start_sample]);
    y_distorted_signal = complex_signal_full(start_sample : start_sample + segment_length - 1);
    
    % --- 还原算法 ---
    filter_window_size = 51; 
    estimated_envelope = smooth(y_distorted_signal, filter_window_size, 'moving');
    epsilon = 1e-9;
    y_envelope_removed = y_distorted_signal ./ (estimated_envelope + epsilon);
    y_centered = y_envelope_removed - mean(y_envelope_removed);

    i_channel_processed = real(y_centered);
    q_channel_processed = imag(y_centered);
    
    min_val_i = min(i_channel_processed);
    i_channel_final = i_channel_processed - min_val_i;
    q_channel_final = q_channel_processed;
    
    y_final_restored = i_channel_final + 1i * q_channel_final;
    
    % --- 步骤 C: 格式化并保存为 .csv 文件 ---
    
    % 【终极稳健修正】：确保 y_final_restored 是行向量
    y_final_restored = y_final_restored(:).'; % (:) 展平为列向量，(') 再转置为行向量
    
    % 现在，我们可以安全地进行垂直拼接
    iq_data_to_save = [real(y_final_restored); imag(y_final_restored)]; % 保证输出 2x1024
    
    % 创建文件名
    filename_csv = fullfile(output_folder, sprintf('restored_ask_sample_%04d.csv', i));
    
    % 使用 writematrix 保存
    writematrix(iq_data_to_save, filename_csv);
    
    if mod(i, 20) == 0 || i == num_files_to_generate
        fprintf('已处理并保存 %d / %d 个文件...\n', i, num_files_to_generate);
    end
end

fprintf('批量还原与保存任务完成！\n');
fprintf('%d 个CSV文件已保存在文件夹 "%s" 中。\n', num_files_to_generate, output_folder);