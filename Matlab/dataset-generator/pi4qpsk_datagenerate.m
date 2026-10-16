clc, clear, close all

%% --- 1. 设置仿真和保存参数 ---
num_samples_per_snr = 6;
snr_range_db = -2:2:36;
output_folder_individual = 'pi4qpsk_200_csv_data'; % 新建一个文件夹
output_data_file = 'test_pi4qpsk_05.mat'; 
output_label_file = 'testlabels_pi4qpsk_05.mat'; 

if ~exist(output_folder_individual, 'dir')
   mkdir(output_folder_individual);
end

total_samples = length(snr_range_db) * num_samples_per_snr;
all_data = cell(total_samples, 1);
all_labels = cell(total_samples, 1);
data_counter = 1;

%% --- 2. 主循环，生成和保存数据 ---
fprintf('开始生成 pi/4-QPSK 数据集...\n');

for snr_db = snr_range_db
    fprintf('正在处理 SNR = %d dB...\n', snr_db);
    
    for sample_idx = 1:num_samples_per_snr
        % --- 核心生成逻辑 ---
        sprintf("num_samples_per_snr: %d",sample_idx)
        RndPhase = true;
        ModType = 'pi4qpsk'; % <-- 关键修改: 指定新的调制类型
        sps_i = 4;
        rolloff = 0.35;
        symbol_order = 'gray';
        % 初始化 pi/4-QPSK 调制器
        [Mod_QPSK] = Initialize_Modulation_NN(ModType, 'sps',sps_i, 'beta',rolloff, 'symbol_order',symbol_order);
        
        if RndPhase
            Mod_QPSK.ini_phase = 2*pi*rand(1,1);
        end

        L = 1024;
        SNR = snr_db;
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        Nsyms = ceil((delay+L+dt+1)/sps_i);
        data = randi([0 1],[1 Nsyms*log2(Mod_QPSK.M)]);
        
        % 使用新的 pi4qpskmod_NN 函数进行调制
        x = pi4qpskmod_NN(Mod_QPSK, data);

        y_noisy = awgn(x, SNR, 'measured');
        y = y_noisy(delay+dt+(1:L));
        % --- 核心生成逻辑结束 ---

        % --- 3. 格式化并保存单条数据为 .csv ---
        iq_data = [real(y); imag(y)]; % 格式为 2x1024

        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0
            snr_str = sprintf('neg%ddB', abs(snr_db));
        end
        mod_name = 'Pi4QPSK'; % <-- 修改标签名称
        
        filename_csv = fullfile(output_folder_individual, ...
            sprintf('%s_%05d_%s.csv', mod_name, data_counter, snr_str));
        
        writematrix(iq_data, filename_csv);

        % --- 4. 存储数据以便后续整合 ---
        all_data{data_counter} = iq_data;
        all_labels{data_counter} = {mod_name, snr_db};
        data_counter = data_counter + 1;
    end
end

%% --- 5. 分别保存数据和标签---
fprintf('所有样本生成完毕。正在整合并保存...\n');

temp_data_matrix = cat(3, all_data{:});
final_data_for_saving = permute(temp_data_matrix, [3, 1, 2]);
final_labels_for_saving = vertcat(all_labels{:});

% --- 保存数据文件 ---
fprintf('正在保存数据文件 "%s"...\n', output_data_file);
data_struct.data_save = final_data_for_saving;
save(output_data_file, '-struct', 'data_struct', '-v7.3');
fprintf('数据文件保存成功！形状为: %s\n', mat2str(size(final_data_for_saving)));

% --- 保存标签文件 ---
fprintf('正在保存标签文件 "%s"...\n', output_label_file);
save(output_label_file, 'final_labels_for_saving', '-v7.3');
fprintf('标签文件保存成功！\n');

fprintf('所有文件已成功创建！\n');