clear; clc; close all;
%% 1. 参数设置
% 假设此时工作区里已经有了 iq_complex (1xN complex single)
% 如果没有，请取消下一行的注释来加载你的大文件
load('IQREC.mat'); 

fs = 40e6;              % 采样率 40MHz
slice_time = 0.002;      % 切片时长 10ms
save_folder = 'IQ_Slices_Data_2'; % 保存切片的文件夹名称

%% 2. 预处理与文件夹创建
% 检查数据是否存在
if ~exist('iq_complex', 'var')
    error('工作区中未找到IQREC变量，请先加载数据！');
end

% 创建保存文件夹 (如果不存在)
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
    fprintf('已创建文件夹: %s\n', save_folder);
else
    fprintf('文件夹已存在: %s，新文件将保存在此。\n', save_folder);
end

%% 3. 核心切片逻辑 (矩阵化处理)
fprintf('正在进行矩阵重塑切片...\n');

slice_len = round(slice_time * fs); % 每个切片的点数 (40MHz * 10ms = 400,000 点)
N = length(iq_complex);

% 计算可切分的完整段数
num_slices = floor(N / slice_len);

if num_slices == 0
    error('数据太短，不足以切出一个完整的 %.2f 秒片段。', slice_time);
end

% 裁剪多余尾部
iq_trimmed = iq_complex(1 : num_slices * slice_len);

% 【核心技巧】 reshape
% 将 1xN 的长条 变形成 slice_len x num_slices 的矩阵
% 每一列(Column) 就是一段完整的切片
sliced_matrix = reshape(iq_trimmed, slice_len, num_slices);

fprintf('数据已重塑为 [%d x %d] 的矩阵。\n', slice_len, num_slices);

%% 4. 批量保存到文件夹
fprintf('开始保存 %d 个文件到 "%s" ...\n', num_slices, save_folder);

% 开启进度条
h = waitbar(0, '正在保存切片数据...');

for k = 1 : num_slices
    % A. 提取第 k 列数据
    % 注意：reshape 后这里是列向量
    temp_col = sliced_matrix(:, k);
    
    % B. 转置为行向量 (符合你之前的"行保存"需求)
    slice_data = temp_col.'; 
    
    % C. 生成文件名 (例如 slice_0001.mat)
    % %04d 表示生成 0001, 0002 这样的序号，方便排序
    filename = sprintf('slice_%04d.mat', k);
    full_path = fullfile(save_folder, filename);
    
    % D. 保存 (包含采样率 fs)
    save(full_path, 'slice_data');
    
    % 更新进度条 (每10个更新一次，防止拖慢速度)
    if mod(k, 10) == 0
        waitbar(k / num_slices, h, sprintf('已保存: %d / %d', k, num_slices));
    end
end

close(h);
fprintf('全部完成！所有文件已保存在文件夹: %s\n', save_folder);