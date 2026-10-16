clc, clear, close all

%16 --> 10
%32 --> 11

%% --- 1. 设置仿真和保存参数 ---
%num_samples_per_snr = 200; %train
num_samples_per_snr = 25; %test
snr_range_db = 18:2:36;  %
output_folder_individual = 'train_200_csv_data_B'; % 文件夹名
%output_folder_individual = 'test_apsk32_individual_csv_data'; % 文件夹名
%output_file_combined = 'train_11.mat'; %train 
output_file_combined = 'test_11.mat'; %test

% 检查并创建单个CSV文件的输出文件夹
if ~exist(output_folder_individual, 'dir')
   mkdir(output_folder_individual);
end

% 初始化用于存储所有数据的cell数组
total_samples = length(snr_range_db) * num_samples_per_snr;
all_data = cell(total_samples, 1);
all_labels = cell(total_samples, 1);
data_counter = 1;

%% --- 2. 主循环，生成和保存数据 ---
fprintf('开始生成APSK数据集...\n');

for snr_db = snr_range_db
    fprintf('正在处理 SNR = %d dB...\n', snr_db);
    
    for sample_idx = 1:num_samples_per_snr
        % --- 核心生成逻辑 (已修改为APSK) ---
        RndPhase = true;
        ModType = 'apsk';   % <-- 修改: 调制类型为APSK
        M_i = 32;           % <-- 修改: 调制阶数, e.g., 16-APSK
        sps_i = 4;
        phaserot = 0;
        ini_phase_i = 0;
        dfcTs_i = 0;
        rolloff = 0.35;
        symbol_order = 'bin'; % 使用二进制映射

        % 初始化APSK调制器
        [Mod_APSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
        
        if RndPhase
            Mod_APSK.ini_phase = 2*pi*rand(1,1);
        end

        L = 1024;
        SNR = snr_db;
        
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        
        Nsyms = ceil((delay+L+dt+1)/sps_i);
        
        % 生成与APSK调制阶数匹配的随机比特
        data = randi([0 1],[1 Nsyms*log2(Mod_APSK.M)]);
        
        % 使用apskmod_NN进行调制
        x = apskmod_NN(Mod_APSK, data); % <-- 修改: 调用apskmod_NN

        %signal_power = mean(abs(x).^2);
        %x_normalized = x / sqrt(signal_power); % 使用幅度的平方根来缩放


        % --- 将归一化后的信号传入 awgn ---
        %y_noisy = awgn(x_normalized, SNR, 'measured'); % 现在awgn在单位功率信号上工作
        y_noisy = awgn(x, SNR, 'measured');
        
        % 截取指定长度的信号
        y = y_noisy(delay+dt+(1:L));
        % --- 核心生成逻辑结束 ---

        % --- 3. 格式化并保存单条数据为 .csv ---
        iq_data = [real(y); imag(y)]; % 格式为 2x1024

        % 创建文件名
        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0
            snr_str = sprintf('neg%ddB', abs(snr_db));
        end
        mod_name = 'APSK32'; % <-- 修改: 标签名称
        
        filename_csv = fullfile(output_folder_individual, ...
            sprintf('%s_%05d_%s.csv', mod_name, data_counter, snr_str));
        
        writematrix(iq_data, filename_csv);

        % --- 4. 存储数据以便后续整合 ---
        all_data{data_counter} = iq_data;
        all_labels{data_counter} = {mod_name, snr_db}; % <-- 修改: 标签内容
        data_counter = data_counter + 1;
    end
end

%% --- 5. 分别保存数据和标签为 .mat 文件 ---

fprintf('所有样本生成完毕。正在整合并保存...\n');

temp_data_matrix = cat(3, all_data{:});
final_data_for_saving = permute(temp_data_matrix, [3, 1, 2]);
final_labels_for_saving = vertcat(all_labels{:});

% --- 定义文件名 ---
%output_data_file = 'train_11.mat';      % <-- 修改: 输出数据文件名
output_data_file = 'test_11.mat';      % <-- 修改: 输出数据文件名
%output_label_file = 'trainlabels_11.mat'; % <-- 修改: 输出标签文件名
output_label_file = 'testlabels_11.mat'; % <-- 修改: 输出标签文件名

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