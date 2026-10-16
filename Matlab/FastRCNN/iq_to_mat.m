clear; clc; close all;

%% 1. 参数设置
input_filename = 'CarFM/IQREC_Carrier98MHz_Fs40Msps_BW27MHz-10-08-24-16h44m33s221.iq';   % 输入的 .iq 文件名
output_filename = 'IQREC.mat'; % 输出的 .mat 文件名
fs = 40e6;                         % 设置采样率 (例如 20MHz)，这将作为一个变量保存

%% 2. 读取二进制文件
% 以二进制只读方式打开
fid = fopen(input_filename, 'rb');

if fid == -1
    error('无法打开文件: %s，请检查路径。', input_filename);
end

disp('正在读取文件...');

% 核心步骤：
% 'int16=>single' 的含义是：源文件按照 int16 (2字节) 读取，
% 但读入内存后直接保存为 single (单精度浮点)，不经过 double 转换，节省内存。
raw_data = fread(fid, Inf, 'int16=>single');

fclose(fid);

%% 3. 数据处理与转换
disp('正在转换数据格式...');

% 检查数据长度是否为偶数 (I和Q必须成对)
num_samples = length(raw_data);
if mod(num_samples, 2) ~= 0
    warning('文件包含奇数个样本点，最后一个样本将被丢弃以保证I/Q对齐。');
    num_samples = num_samples - 1;
    raw_data = raw_data(1:num_samples);
end

% 提取 I 和 Q 分量
% 假设排列顺序为 I, Q, I, Q ...
I_data = raw_data(1:2:end);
Q_data = raw_data(2:2:end);

% 组合成复数 (I + jQ)
% 由于 I_data 和 Q_data 已经是 single 类型，结果自动就是 complex single
iq_complex = complex(I_data, Q_data);
iq_complex = iq_complex.';
% 清理原始数据以释放内存
clear raw_data I_data Q_data;

%% 4. 保存为 .mat 文件
disp(['正在保存为 ', output_filename, ' ...']);

% 保存 iq_complex 数据和 fs 采样率
% '-v7.3' 格式支持大于 2GB 的文件，建议加上以防数据过大
save(output_filename, 'iq_complex', 'fs', '-v7.3');

disp('处理完成！');

%% 5. (可选) 验证数据格式
disp('--- 结果验证 ---');
whos_info = whos('iq_complex');
fprintf('变量名: %s\n', whos_info.name);
fprintf('数据类型: %s\n', whos_info.class); % 应该显示 'single'
fprintf('是否复数: %s\n', mat2str(whos_info.complex)); % 应该显示 true