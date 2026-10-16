clc, clear, close all

%% --- 1. 设置仿真、保存和干扰参数 ---

% --- 数据集参数 ---
num_samples_per_snr = 200;
snr_range_db = -2:2:36;
output_folder_individual = 'train_pi4qpsk_impairments_csv'; % 新建文件夹
output_data_file = 'train_pi4qpsk_impairments_09.mat'; 
output_label_file = 'trainlabels_pi4qpsk_impairments_09.mat'; 

% --- 干扰参数 ---
impairment_probabilities = struct(...
    'cfo', 0.5, ...       % 50% 的样本会添加CFO
    'sco', 0.5, ...       % 50% 的样本会添加SCO
    'iq_imbalance', 0.4, ... % 40% 的样本会添加IQ不平衡
    'fading', 0.3 ...     % 30% 的样本会通过多径衰落信道
);

impairment_strengths = struct(...
    'max_cfo_norm', 0.05, ...         % 最大归一化频偏 (符号速率的百分比)
    'max_sco_ppm', 50, ...            % 最大时钟偏移 (百万分率, ppm)
    'max_amp_imb_dB', 1.0, ...        % 最大幅度不平衡 (dB)
    'max_phase_imb_deg', 8, ...       % 最大相位不平衡 (度)
    'max_doppler_shift', 5 ...        % 最大多普勒频移 (Hz)
);


if ~exist(output_folder_individual, 'dir')
   mkdir(output_folder_individual);
end

total_samples = length(snr_range_db) * num_samples_per_snr;
all_data = cell(total_samples, 1);
all_labels = cell(total_samples, 1);
data_counter = 1;

%% --- 2. 初始化信道对象 (为提高效率，在循环外创建) ---
% !! 需要 Communications Toolbox !!
sps_i = 4; % 必须在这里定义，因为信道对象需要采样率
fs = sps_i; 
fading_channel = comm.RayleighChannel(...
    'SampleRate', fs, ...
    'PathDelays', [0 0.25 0.5] / fs, ... % 简单的三径模型，延迟为0, 0.25, 0.5个符号周期
    'AveragePathGains', [0 -3 -6], ...   % 平均路径增益 (dB)
    'MaximumDopplerShift', impairment_strengths.max_doppler_shift, ...
    'DopplerSpectrum', doppler('Jakes'));


%% --- 3. 主循环，生成和保存数据 ---
fprintf('开始生成带损伤的 pi/4-QPSK 数据集...\n');

for snr_db = snr_range_db
    fprintf('正在处理 SNR = %d dB...\n', snr_db);
    
    for sample_idx = 1:num_samples_per_snr
        % --- 3.1 核心信号生成逻辑 (保持不变) ---
        RndPhase = true;
        ModType = 'pi4qpsk'; 
        rolloff = 0.35;
        symbol_order = 'gray';
        [Mod_QPSK] = Initialize_Modulation_NN(ModType, 'sps',sps_i, 'beta',rolloff, 'symbol_order',symbol_order);
        
        if RndPhase
            Mod_QPSK.ini_phase = 2*pi*rand(1,1);
        end

        L = 1024;
        SNR = snr_db;
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        Nsyms = ceil((delay+L+dt+1)/sps_i) + 8; % 增加一些富余的符号以应对滤波器和重采样
        data = randi([0 1],[1 Nsyms*log2(Mod_QPSK.M)]);
        
        x = pi4qpskmod_NN(Mod_QPSK, data);
        
        % --- 3.2 随机施加各种损伤 ---
        current_x = x; % 从原始无损信号开始
        
        % --- 采样时钟偏移 (SCO) ---
        if rand < impairment_probabilities.sco
            sco_ppm = (rand*2-1) * impairment_strengths.max_sco_ppm;
            sco_factor = 1 + sco_ppm / 1e6;
            
            original_time = 1:length(current_x);
            resampled_time = linspace(1, length(current_x), round(length(current_x) / sco_factor)); % 注意是除法
            current_x = interp1(original_time, current_x, resampled_time, 'spline');
        end

        % --- 载波频率偏移 (CFO) ---
        if rand < impairment_probabilities.cfo
            cfo_norm = (rand*2-1) * impairment_strengths.max_cfo_norm;
            cfo_phase_inc = 2 * pi * cfo_norm / sps_i;
            time_indices = 0:(length(current_x)-1);
            cfo_phasor = exp(1i * cfo_phase_inc * time_indices);
            current_x = current_x .* cfo_phasor;
        end

        % --- IQ不平衡 ---
        if rand < impairment_probabilities.iq_imbalance
            amp_imb_dB = rand * impairment_strengths.max_amp_imb_dB;
            phase_imb_deg = (rand*2-1) * impairment_strengths.max_phase_imb_deg;

            g = 10^(amp_imb_dB / 20);
            phi = phase_imb_deg * (pi/180);
            
            I = real(current_x);
            Q = imag(current_x);
            current_x = (g * I) + 1i * (I * tan(phi) + Q / cos(phi));
        end

        % --- 多径瑞利衰落 ---
        % !! 需要 Communications Toolbox !!
        % (如果报错，请注释掉这个 if-end 代码块)
        if rand < impairment_probabilities.fading
            faded_x = fading_channel(current_x(:));
            current_x = faded_x.';
            release(fading_channel); % 重置信道状态，确保每个样本独立
        end

        % --- 3.3 添加噪声并截取最终样本 ---
        y_noisy = awgn(current_x, SNR, 'measured');
        
        % 确保有足够的样本长度可以截取
        if length(y_noisy) >= (delay + dt + L)
            y = y_noisy(delay+dt+(1:L));
        else
            % 如果信号因重采样变短，用零填充
            y = [y_noisy(delay+dt+1:end), zeros(1, delay+dt+L - length(y_noisy))];
        end

        % --- 3.4 格式化并保存 ---
        iq_data = [real(y); imag(y)];

        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0, snr_str = sprintf('neg%ddB', abs(snr_db)); end
        mod_name = 'Pi4QPSK';
        
        filename_csv = fullfile(output_folder_individual, ...
            sprintf('%s_%05d_%s.csv', mod_name, data_counter, snr_str));
        writematrix(iq_data, filename_csv);

        all_data{data_counter} = iq_data;
        all_labels{data_counter} = {mod_name, snr_db};
        data_counter = data_counter + 1;
    end
end

%% --- 4. 整合并保存数据集 (保持不变) ---
fprintf('所有样本生成完毕。正在整合并保存...\n');

temp_data_matrix = cat(3, all_data{:});
final_data_for_saving = permute(temp_data_matrix, [3, 1, 2]);
final_labels_for_saving = vertcat(all_labels{:});

fprintf('正在保存数据文件 "%s"...\n', output_data_file);
data_struct.data_save = final_data_for_saving;
save(output_data_file, '-struct', 'data_struct', '-v7.3');
fprintf('数据文件保存成功！形状为: %s\n', mat2str(size(final_data_for_saving)));

fprintf('正在保存标签文件 "%s"...\n', output_label_file);
save(output_label_file, 'final_labels_for_saving', '-v7.3');
fprintf('标签文件保存成功！\n');

fprintf('所有文件已成功创建！\n');