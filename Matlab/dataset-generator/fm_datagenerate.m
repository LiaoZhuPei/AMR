% FM 数据集生成脚本
% 核心逻辑使用 FM 调制

clc, clear, close all

%% --- 1. 设置仿真和保存参数 ---
%train - 200 ; test - 25
num_samples_per_snr = 25;
snr_range_db = -2:2:36;
output_folder_individual = 'train_200_csv_data_FM'; % 单独为FM创建一个文件夹
output_data_file = 'test_15.mat';      % 建议为不同调制方式使用不同文件名
output_label_file = 'testlabels_15.mat';

if ~exist(output_folder_individual, 'dir')
   mkdir(output_folder_individual);
end

total_samples = length(snr_range_db) * num_samples_per_snr;
all_data = cell(total_samples, 1);
all_labels = cell(total_samples, 1);
data_counter = 1;

%% --- 2. 主循环，生成和保存数据 ---
fprintf('开始生成FM数据集...\n');

for snr_db = snr_range_db
    fprintf('正在处理 SNR = %d dB...\n', snr_db);
    
    for sample_idx = 1:num_samples_per_snr
        % --- 核心生成逻辑 (替换为 FM) ---
        
        % --- FM 调制器参数 ---
        RndPhase = true;        % 是否使用随机初始相位
        ModType = 'fm';
        h_i = 0.2;              % 调制指数 (β or h)。当前为窄带FM (NBFM)
        sps_i = 4;              % 上采样因子
        ini_phase_i = 0;        % 默认初始相位
        
        % --- 基带和载波频率参数 ---
        fs_baseband = 6e3;      % 基带信号的采样率 (Hz)
        fc = 5e3;               % 载波频率 (Hz)
        fs_rf = fs_baseband * sps_i; % 调制后信号的采样率
        dfc_div_fs_i = fc / fs_rf;   % 归一化的载波频率

        % 初始化FM调制器结构体
        [Mod_FM] = Initialize_Modulation_NN(ModType, 'h', h_i, 'sps', sps_i, ...
            'ini_phase', ini_phase_i, 'dfc_div_fs', dfc_div_fs_i);

        if RndPhase
            Mod_FM.ini_phase = 2 * pi * rand(1, 1);
        end
        
        % --- 生成随机化的基带信号 (关键步骤) ---
        L = 1024; % 最终输出的射频信号长度
        
        % 引入随机时间偏移，与OQPSK脚本保持一致
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        
        % 计算需要的基带信号长度
        L_baseband_needed = ceil((delay + L + dt + 1) / sps_i);
        
        % 创建随机基带信号 (与AM脚本生成方式相同，保证一致性)
        % 1. 生成高斯白噪声
        raw_signal = randn(1, L_baseband_needed + 100); 
        % 2. 设计低通滤波器
        filter_order = 6;
        nyquist_freq = fs_baseband / 2;
        cutoff_freq = nyquist_freq * 0.8; 
        [b, a] = butter(filter_order, cutoff_freq/nyquist_freq, 'low');
        % 3. 滤波
        filtered_signal = filter(b, a, raw_signal);
        m_t_raw = filtered_signal(101:end); 
        % 4. 归一化信号，使其峰值在 [-1, 1] 之间
        m_t = m_t_raw / max(abs(m_t_raw)); 

        % 调用调制函数生成理想的FM信号
        x = fmmod_NN(Mod_FM, m_t);
      
        % 通过AWGN信道添加噪声
        SNR = snr_db;
        y_noisy = awgn(x, SNR, 'measured');

        % 截取最终长度为 L 的信号，并应用随机时间偏移
        y = y_noisy(delay + dt + (1:L));
        % --- 核心生成逻辑结束 ---

        % --- 3. 格式化并保存单条数据为 .csv ---
        iq_data = [real(y); imag(y)]; % 格式为 2x1024

        % 创建文件名
        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0
            snr_str = sprintf('neg%ddB', abs(snr_db));
        end
        mod_name = 'FM'; % <--- 修改调制名称
        filename_csv = fullfile(output_folder_individual, ...
            sprintf('%s_%05d_%s.csv', mod_name, data_counter, snr_str));
        
        % 使用 writematrix 保存为 .csv 文件
        writematrix(iq_data, filename_csv);

        % --- 4. 存储数据以便后续整合 ---
        all_data{data_counter} = iq_data;
        all_labels{data_counter} = {mod_name, snr_db}; % <--- 修改标签
        data_counter = data_counter + 1;
    end
end

%% --- 5. 分别保存数据和标签 (与OQPSK/AM脚本完全相同) ---

fprintf('所有样本生成完毕。正在整合并保存...\n');

% 将 cell 数组整合成一个 2x1024xN 的矩阵
temp_data_matrix = cat(3, all_data{:});
% 转置为 N x 2 x 1024
final_data_for_saving = permute(temp_data_matrix, [3, 1, 2]);

% 整合标签
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