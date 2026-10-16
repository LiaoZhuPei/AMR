clc, clear, close all

%% --- 1. 设置仿真和保存参数 ---

% =========================================================================
% --- 关键设置：生成训练集还是测试集？ ---
is_test_set = false; % <--- 设置为 true 可生成测试集
% =========================================================================

if is_test_set
    num_samples_per_snr = 50;      % 测试集的样本数量
    output_folder_individual = 'test_50_csv_data_ASK2_B';
    output_data_file = 'test_08.mat';
    output_label_file = 'testlabels_08.mat';
else
    num_samples_per_snr = 400;     % 训练集的样本数量 (推荐增加)
    output_folder_individual = 'train_400_csv_data_ASK2_B';
    output_data_file = 'train_08.mat';
    output_label_file = 'trainlabels_08.mat';
end

snr_range_db = -2:2:36;

% --- 新增参数：用于方法B的信号尺寸 ---
MAX_LEN = 500;            % 最终网络输入的固定长度 (画布大小)
LONG_SIG_LEN = 2048;        % 用于截取的长信号流的长度 (信号之河)
MIN_SIGNAL_LEN = 100;       % 截取信号的最小长度
MAX_SIGNAL_LEN = 500;       % 截取信号的最大长度 (允许占满画布)

if ~exist(output_folder_individual, 'dir')
   mkdir(output_folder_individual);
end

total_samples = length(snr_range_db) * num_samples_per_snr;
all_data = cell(total_samples, 1);
all_labels = cell(total_samples, 1);
data_counter = 1;

%% --- 2. 主循环，生成和保存数据 ---
fprintf('开始生成 ASK 数据集 (方法B)...\n');
if is_test_set, fprintf('模式: 测试集\n'); else, fprintf('模式: 训练集\n'); end

for snr_db = snr_range_db
    fprintf('正在处理 SNR = %d dB...\n', snr_db);
    
    for sample_idx = 1:num_samples_per_snr
        % =================================================================
        % --- 核心生成逻辑 (已按方法B修改) ---
        % =================================================================
        
        % --- a. 生成一个长信号流 (信号之河) ---
        RndPhase = true;
        ModType = 'ask'; 
        M_i = 2;            
        sps_i = 4;
        phaserot = 0;
        ini_phase_i = 0;
        dfcTs_i = 0;
        rolloff = 0.35;
        symbol_order = 'gray';

        [Mod_ASK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
        if RndPhase
            Mod_ASK.ini_phase = 2*pi*rand(1,1);
        end

        SNR = snr_db;
        
        Nsyms_long = ceil(LONG_SIG_LEN / sps_i);
        data_long = randi([0 1],[1 Nsyms_long*log2(Mod_ASK.M)]);
        x_long = askmod_NN(Mod_ASK, data_long);
        
        % --- b. 从长信号流中随机截取一个可变长度的片段 ---
        signal_len = randi([MIN_SIGNAL_LEN, MAX_SIGNAL_LEN]);
        
        if length(x_long) > signal_len
            crop_start = randi([1, length(x_long) - signal_len + 1]);
            x_cropped = x_long(crop_start : crop_start + signal_len - 1);
        else
            x_cropped = x_long;
            signal_len = length(x_long);
        end

        % --- c. 创建画布并将截取的信号随机放置上去 ---
        canvas = zeros(1, MAX_LEN, 'like', 1i);
        start_pos = randi([1, MAX_LEN - signal_len + 1]);
        canvas(start_pos : start_pos + signal_len - 1) = x_cropped;

        % --- d. 对整个画布加入噪声 ---
        canvas_noisy = awgn(canvas, SNR, 'measured');
        
        % --- e. 对带噪信号加入随机DC偏置 (模拟接收机损伤) ---
        i_offset_rand = -0.5 + (-0.3 - (-0.5)) * rand();
        q_offset_rand = 0.3 + (0.5 - 0.3) * rand();
        dc_offset = i_offset_rand + 1j * q_offset_rand;
        
        y_final = canvas_noisy + dc_offset;
        
        % =================================================================
        % --- 核心生成逻辑结束 ---
        % =================================================================

        % --- 3. 格式化并保存单条数据为 .csv ---
        iq_data = [real(y_final); imag(y_final)]; % 格式为 2x500

        % 创建文件名
        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0
            snr_str = sprintf('neg%ddB', abs(snr_db));
        end
        mod_name = 'ASK2';
        
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

% 将 cell 数组整合成一个 2x500xN 的矩阵
temp_data_matrix = cat(3, all_data{:});
% 转置为 N x 2 x 500
final_data_for_saving = permute(temp_data_matrix, [3, 1, 2]);

% 整合标签
final_labels_for_saving = vertcat(all_labels{:});

% --- 保存数据文件 (文件名已在顶部定义) ---
fprintf('正在保存数据文件 "%s"...\n', output_data_file);
data_struct.data_save = final_data_for_saving;
save(output_data_file, '-struct', 'data_struct', '-v7.3');
fprintf('数据文件保存成功！形状为: %s\n', mat2str(size(final_data_for_saving)));

% --- 保存标签文件 (文件名已在顶部定义) ---
fprintf('正在保存标签文件 "%s"...\n', output_label_file);
save(output_label_file, 'final_labels_for_saving', '-v7.3');
fprintf('标签文件保存成功！\n');

fprintf('所有文件已成功创建！\n');