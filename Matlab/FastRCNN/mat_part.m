clear; clc; close all;

% 1.设置
source_file = '20170515_Official.mat';
output_prefix = '20170515_PartD_';  %原版10s，B-5s，C-1s, D-0.6s
segment_time = 0.6;  % 设定每段的长度（秒）
fs = 1000000;       % 采样率

% 2.加载原始文件
fprintf('正在加载长文件: %s...\n', source_file);
load(source_file, 'signal');

% 检查是否为 1*N
if size(signal, 1) > size(signal, 2)
    error('格式错误');
end

total_len = length(signal);
segment_len = round(segment_time * fs); % 每段的数据点数
num_segments = ceil(total_len / segment_len); % 计算总段数

fprintf('总长度: %d 点, 将分为 %d 个文件保存 (每段 %.1f 秒)\n', total_len, num_segments, segment_time);

% 3.循环切分并保存
for i = 1:num_segments
    idx_start = (i-1) * segment_len + 1;
    idx_end = min(i * segment_len, total_len);

    signal_segment = signal(idx_start : idx_end);

    out_filename = sprintf('%s%d.mat', output_prefix, i);

    data_to_save.signal = signal_segment;

    save(out_filename, '-struct', 'data_to_save');

    fprintf('已保存: %s (点数：%d)\n', out_filename, length(signal_segment));
end

fprintf('全部完成！\n');
