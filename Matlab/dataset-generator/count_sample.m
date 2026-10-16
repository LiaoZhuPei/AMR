clc, clear, close all

%% --- 1. 设置仿真和保存参数 ---
num_samples_per_snr = 25;
snr_range_db = -2:2:36;
output_folder_individual = 'individual_data';
output_file_combined = 'HisarMod_like_dataset.mat';

% --- 新增：自动计算并显示总样本数 ---
num_snr_levels = length(snr_range_db);
total_samples = num_samples_per_snr * num_snr_levels;

% 使用 fprintf 打印信息到命令行
fprintf('============================================\n');
fprintf('       数据集生成配置信息\n');
fprintf('============================================\n');
fprintf('每个信噪比下的样本数: %d\n', num_samples_per_snr);
fprintf('信噪比等级数量: %d\n', num_snr_levels);
fprintf('信噪比范围: 从 %d dB 到 %d dB\n', snr_range_db(1), snr_range_db(end));
fprintf('--------------------------------------------\n');
fprintf('>>> 将要生成的总样本数: %d\n', total_samples);
fprintf('============================================\n\n');