%增加了非线性失真
clc, clear, close all

%% --- 1. 设置仿真和保存参数 ---
%num_samples_per_snr = 200;  %Train
num_samples_per_snr = 25;  %Test
snr_range_db = -2:2:36;

output_folder_individual = 'QAM_Rapp_csv_data'; % 为区分，建议文件夹名也体现内容

%output_file_combined = 'train_07.mat';  %16QAM
%output_file_combined = 'train_12.mat'; %32QAM
%output_file_combined = 'train_13.mat'; %64QAM

%output_file_combined = 'test_07.mat';
%output_file_combined = 'test_12.mat';
output_file_combined = 'test_13.mat';

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
        % --- 核心生成逻辑 ---
        % (这部分参数保持不变)
        RndPhase = true;
        ModType = 'qam';
        M_i = 64;
        sps_i = 4;
        phaserot = 0;
        ini_phase_i = 0;
        dfcTs_i = 0;
        rolloff = 0.35;
        symbol_order = 'gray';
        [Mod_QAM] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
        if RndPhase
            Mod_QAM.ini_phase = 2*pi*rand(1,1);
        end
        L = 1024;
        SNR = snr_db;
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        Nsyms = ceil((delay+L+dt+1)/sps_i);
        data = randi([0 1],[1 Nsyms*log2(Mod_QAM.M)]);
        
        % 生成原始的、无失真的信号
        x_linear = qammod_NN(Mod_QAM, data);

        % =====================================================================
        % <<< --- 修改点：在此处为每个样本添加随机非线性失真 --- >>>
        % =====================================================================
        % 定义Rapp模型参数的随机范围
        p_min = 1.8; p_max = 2.5; % 平滑因子范围
        Asat_min = 0.9; Asat_max = 1.2; % 饱和幅度范围 (相对于峰值幅度)
        
        % 为当前样本随机生成一组参数
        p_rand = p_min + (p_max - p_min) * rand();
        Asat_rand = Asat_min + (Asat_max - Asat_min) * rand();
        
        % 峰值幅度归一化
        x_peak_power = max(abs(x_linear).^2);
        x_normalized_for_pa = x_linear / sqrt(x_peak_power);
        
        % 应用Rapp模型
        input_amp = abs(x_normalized_for_pa);
        output_amp = input_amp ./ (1 + (input_amp / Asat_rand).^(2*p_rand)).^(1/(2*p_rand));
        
        % 得到失真信号
        x_distorted = x_normalized_for_pa .* (output_amp ./ input_amp);
        
        % 功率恢复，以进行公平的SNR比较
        power_linear = mean(abs(x_linear).^2);
        power_distorted_temp = mean(abs(x_distorted).^2);
        x_nonlinear = x_distorted * sqrt(power_linear / power_distorted_temp);
        
        % 现在，x_nonlinear 是我们送入信道的最终信号
        % =====================================================================

        % 将失真后的信号送入AWGN信道
        y_noisy = awgn(x_nonlinear, SNR, 'measured');
        
        % 截取突发信号
        y = y_noisy(delay+dt+(1:L));
        % --- 核心生成逻辑结束 ---

        % --- 3. 格式化并保存单条数据为 .csv ---
        iq_data = [real(y); imag(y)]; % 格式为 2x1024
        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0
            snr_str = sprintf('neg%ddB', abs(snr_db));
        end
        mod_name = 'QAM64';
        filename_csv = fullfile(output_folder_individual, ...
            sprintf('%s_%05d_%s.csv', mod_name, data_counter, snr_str));
        writematrix(iq_data, filename_csv);

        % --- 4. 存储数据以便后续整合 ---
        all_data{data_counter} = iq_data;
        all_labels{data_counter} = {mod_name, snr_db};
        data_counter = data_counter + 1;
    end
end

%% --- 5. 分别保存数据和标签 ---
% (这部分代码保持不变)
fprintf('所有样本生成完毕。正在整合并保存...\n');

temp_data_matrix = cat(3, all_data{:});
final_data_for_saving = permute(temp_data_matrix, [3, 1, 2]);
final_labels_for_saving = vertcat(all_labels{:});

% --- 定义文件名 ---
%output_data_file = 'train_07.mat';
%output_label_file = 'trainlabels_07.mat';
%output_data_file = 'train_12.mat';
%output_label_file = 'trainlabels_12.mat';
%output_data_file = 'train_13.mat';
%output_label_file = 'trainlabels_13.mat';

%output_data_file = 'test_07.mat';
%output_label_file = 'testlabels_07.mat';
%output_data_file = 'test_12.mat';
%output_label_file = 'testlabels_12.mat';
output_data_file = 'test_13.mat';
output_label_file = 'testlabels_13.mat';

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