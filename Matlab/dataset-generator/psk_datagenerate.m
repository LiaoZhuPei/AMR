clc, clear, close all

%% --- 1. 设置仿真和保存参数 ---
num_samples_per_snr = 200;
snr_range_db = -2:2:36;
output_folder_individual = 'train_200_csv_data_A'; % 文件夹名改为csv
output_file_combined = 'train_03.mat'; 

if ~exist(output_folder_individual, 'dir')
   mkdir(output_folder_individual);
end

total_samples = length(snr_range_db) * num_samples_per_snr;
all_data = cell(total_samples, 1);
all_labels = cell(total_samples, 1);
data_counter = 1;

%% --- 2. 主循环，生成和保存数据 ---
fprintf('开始生成数据集...\n');

for snr_db = snr_range_db
    fprintf('正在处理 SNR = %d dB...\n', snr_db);
    
    for sample_idx = 1:num_samples_per_snr
        % --- 核心生成逻辑 (保持不变) ---
        RndPhase = true;
        ModType = 'psk';
        M_i = 8;
        sps_i = 4;
        phaserot = 0;
        ini_phase_i = 0;
        dfcTs_i = 0;
        rolloff = 0.35;
        symbol_order = 'gray';
        [Mod_8PSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
        if RndPhase
            Mod_8PSK.ini_phase = 2*pi*rand(1,1);
        end
        L = 1024;
        SNR = snr_db;
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        Nsyms = ceil((delay+L+dt+1)/sps_i);
        data = randi([0 1],[1 Nsyms*log2(Mod_8PSK.M)]);
        x = pskmod_NN(Mod_8PSK, data);

        y_noisy = awgn(x, SNR, 'measured');

        %y_noisy = awgn(x, SNR, 'measured');
        y = y_noisy(delay+dt+(1:L));
        % --- 核心生成逻辑结束 ---

        % --- 3. 格式化并保存单条数据为 .csv ---
        iq_data = [real(y); imag(y)]; % 格式为 2x1024

        % 创建文件名，后缀改为 .csv
        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0
            snr_str = sprintf('neg%ddB', abs(snr_db));
        end
        mod_name = '8PSK';
        % 注意文件后缀已改为 .csv
        filename_csv = fullfile(output_folder_individual, ...
            sprintf('%s_%05d_%s.csv', mod_name, data_counter, snr_str));
        
        % !! 关键修改：使用 writematrix 保存为 .csv 文件 !!
        writematrix(iq_data, filename_csv);

        % --- 4. 存储数据以便后续整合 ---
        all_data{data_counter} = iq_data;
        all_labels{data_counter} = {mod_name, snr_db};
        data_counter = data_counter + 1;
    end
end

%% --- 5. 分别保存数据和标签---

fprintf('所有样本生成完毕。正在整合并保存...\n');

% 将 cell 数组整合成一个 2x1024xN 的矩阵
temp_data_matrix = cat(3, all_data{:});
% 转置为 N x 2 x 1024，这是深度学习中常见的格式
final_data_for_saving = permute(temp_data_matrix, [3, 1, 2]);

% 整合标签
final_labels_for_saving = vertcat(all_labels{:});

% --- 定义文件名 ---
output_data_file = 'train_03.mat';
output_label_file = 'trainlabels_03.mat'; 

% --- 保存数据文件 ---
fprintf('正在保存数据文件 "%s"...\n', output_data_file);
data_struct.data_save = final_data_for_saving;
save(output_data_file, '-struct', 'data_struct', '-v7.3');
fprintf('数据文件保存成功！形状为: %s\n', mat2str(size(final_data_for_saving)));

% --- 保存标签文件 ---
fprintf('正在保存标签文件 "%s"...\n', output_label_file);
% 直接保存，文件中的变量名就是 'final_labels_for_saving'
save(output_label_file, 'final_labels_for_saving', '-v7.3');
fprintf('标签文件保存成功！\n');

fprintf('所有文件已成功创建！\n');