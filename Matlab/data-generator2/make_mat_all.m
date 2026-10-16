clearvars;
clc;
close all;

% --- 1. 设置路径和文件模式 ---
data_folder = 'E:/M_Project/Part_IQ_500/MatA2/train';

file_pattern = fullfile(data_folder, 'train_*.mat');
%file_pattern = fullfile(data_folder, 'test_*.mat');

% 查找所有匹配的文件
file_structs = dir(file_pattern);

% 检查是否找到了文件
if isempty(file_structs)
    error("在文件夹 '%s' 中没有找到任何匹配 '%s' 的文件。", data_folder, file_pattern);
end

% 从结构体中提取文件名并排序，确保 train_10 在 train_2 之后
file_names = {file_structs.name};

fprintf('找到 %d 个数据文件，将按以下顺序合并：\n', numel(file_names));
for i = 1:numel(file_names)
    fprintf('  - %s\n', file_names{i});
end

% --- 2. 加载并收集所有数据块 ---
all_data_cell = {}; % 使用元胞数组存储每个处理后的数据块

fprintf('\n开始加载和处理数据...\n');
for i = 1:numel(file_names)
    current_filename = file_names{i};
    full_path = fullfile(data_folder, current_filename);
    
    fprintf('正在处理: %s...\n', current_filename);
    
    % 加载 .mat 文件
    loaded_content = load(full_path, 'data_save');
    
    if ~isfield(loaded_content, 'data_save')
        error("错误! 文件 '%s' 中没有找到名为 'data_save' 的变量。", current_filename);
    end
    
    data_original_shape = loaded_content.data_save;

    all_data_cell{end+1} = data_original_shape;
    
    fprintf('  > 原始形状: (%s) \n', strjoin(string(size(data_original_shape)), ', '));
end


% --- 3. 拼接并保存 ---
disp(newline); % 打印一个新行
disp('正在内存中拼接所有数据...');

% 沿着第 1 个维度（样本维度）拼接所有数据块
% 这将把 (N1, 2, 1024), (N2, 2, 1024), ... 合并为 (N1+N2+..., 2, 1024)
combined_data = cat(1, all_data_cell{:});

fprintf('数据拼接完成！最终数据形状: (%s)\n', strjoin(string(size(combined_data)), ', '));

% 为了与Python脚本的输出保持一致，我们将合并后的数据保存到
% 一个名为 'data_save' 的变量中。
data_save = combined_data;

output_mat_path = fullfile(data_folder, 'train.mat');
%output_mat_path = fullfile(data_folder, 'test.mat');
fprintf('\n正在将数据保存到 .mat 文件: %s\n', output_mat_path);

% 使用 -v7.3 标志来支持大文件
save(output_mat_path, 'data_save', '-v7.3');

disp('  > .mat 文件保存成功！');

disp(newline);
disp('所有操作完成！');