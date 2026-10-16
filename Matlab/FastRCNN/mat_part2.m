clear; clc; close all;

%将复数保存的.mat文件按时间长度划分为小片段
%% 1.设置
source_file = '20170515_Official.mat';
output_prefix = '20170515_PartG_';
fs = 1000000;

% 时间设置
time_first = 1.06;  % 第一段时间
time_rest = 0.71;   % 后续段时间

%% 2.加载原始文件
fprintf('正在加载长文件：%s...\n', source_file);
load(source_file, 'signal');

% 检查格式
if size(signal, 1) > size(signal, 2)
    error('格式错误');
end

total_len = length(signal);
len_first = round(time_first * fs);
len_rest = round(time_rest * fs);

fprintf('总长度：%d 点\n', total_len);
fprintf('切分策略: 第1个文件 %.1fs, 后续文件 %.1fs\n', time_first, time_rest);

%% 3.处理第1个片段（0.4s）
idx_start = 1;
idx_end = min(len_first, total_len);

% 提取
signal_segment = signal(idx_start: idx_end);

% 保存
out_filename = sprintf('%s1.mat', output_prefix);
data_to_save.signal = signal_segment;
save(out_filename, '-struct', 'data_to_save');

fprintf('已保存: %s (点数：%d, 时长: %.2fs)\n', out_filename, length(signal_segment), length(signal_segment)/fs);

%% 4.循环处理剩余部分（0.6s）
if total_len > len_first
    remaining_start_idx = len_first + 1;

    remaining_len = total_len - len_first;
    num_rest_segments = ceil(remaining_len/len_rest);

    for i = 1 : num_rest_segments
        curr_start = len_first + (i-1) * len_rest + 1;
        curr_end = min(curr_start + len_rest - 1, total_len);

        signal_segment = signal(curr_start : curr_end);

        out_filename = sprintf('%s%d.mat', output_prefix, i + 1);

        data_to_save.signal = signal_segment;

        save(out_filename, '-struct', 'data_to_save');

        fprintf('已保存: %s (点数：%d, 时长: %.2fs)\n', ...
                out_filename, length(signal_segment), length(signal_segment)/fs);
    end
end

fprintf('全部完成！\n');