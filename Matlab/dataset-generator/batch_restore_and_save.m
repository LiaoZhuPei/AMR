% =========================================================================
%                batch_restore_and_save.m
%       (批量生成失真信号，并将其还原后保存为CSV文件)
% =========================================================================
clc, clear, close all

%% --- 1. 用户配置区 ---
num_files_to_generate = 100; % 要生成和还原的文件数量
output_folder = 'restored_ask_signals_csv'; % 保存CSV文件的文件夹

% 检查并创建输出文件夹
if ~exist(output_folder, 'dir')
   mkdir(output_folder);
end

fprintf('开始批量生成和还原 %d 个信号...\n', num_files_to_generate);

%% --- 2. 主循环 ---
for i = 1:num_files_to_generate
    
    % --- 步骤 A: 生成一个带随机初始相位的高保真失真信号 ---
    % (这部分代码与我们之前的仿真器完全相同，但关键的 initial_phase_offset_rad 是随机的)
    SimulateDistortion = true; 
    amplitude_on = 1.0; amplitude_off = 0.36; 
    total_rotations_in_frame = 0.6;
    initial_phase_offset_rad = rand() * 2 * pi; % 【核心】每次都生成随机相位
    dc_offset_i = -0.5; dc_offset_q = -0.5;
    global_gain = 3000; SNR = 25; L = 1024; sps_i = 4;
    ModType = 'ask'; M_i = 2;
    [Mod_ASK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i);
    filter_delay = Mod_ASK.PulseShaping.delay;
    num_required_samples = L + sps_i;
    Nsyms = ceil(num_required_samples / sps_i) + ceil(filter_delay * 2 / sps_i);
    data = randi([0 1],[1 Nsyms*log2(M_i)]);
    x_raw_filtered = askmod_NN(Mod_ASK, data);
    map_on = (x_raw_filtered == 1); map_off = (x_raw_filtered == -1);
    x_imperfect_ook = zeros(size(x_raw_filtered));
    x_imperfect_ook(map_on) = amplitude_on; x_imperfect_ook(map_off) = amplitude_off;
    x_distorted = x_imperfect_ook;
    if SimulateDistortion
        normalized_frequency = total_rotations_in_frame / L;
        n = 0:(length(x_imperfect_ook)-1);
        quadratic_phase = 0.000001 * n.^2; 
        cfo_rotation_factor = exp(1i * (2 * pi * normalized_frequency * n + initial_phase_offset_rad + quadratic_phase));
        x_distorted = x_imperfect_ook .* cfo_rotation_factor;
        x_distorted = x_distorted + (dc_offset_i + 1i * dc_offset_q);
    end
    signal_power = mean(abs(x_distorted(filter_delay+1:end-filter_delay)).^2);
    snr_linear = 10^(SNR / 10); noise_power = signal_power / snr_linear;
    noise_std_dev = sqrt(noise_power / 2);
    complex_noise = noise_std_dev * (randn(size(x_distorted)) + 1i*randn(size(x_distorted)));
    y_noisy = x_distorted + complex_noise;
    start_index = filter_delay + 1; end_index = start_index + L - 1;
    y_final_burst = y_noisy(start_index:end_index);
    y_final_burst = y_final_burst * global_gain;
    y_final_burst = conj(y_final_burst);
    y_distorted_signal = y_final_burst;

    % --- 步骤 B: 对生成的失真信号进行逆向还原 ---
    % (这部分代码是我们最终的、最稳健的还原算法)
    y_step1 = conj(y_distorted_signal);
    rms_val = sqrt(mean(abs(y_step1).^2));
    y_step2 = y_step1 / rms_val;
    dc_offset_estimated = mean(y_step2);
    y_step3 = y_step2 - dc_offset_estimated;
    
    phase_all = angle(y_step3);
    unwrapped_phase_all = unwrap(phase_all);
    amp_threshold = (max(abs(y_step3)) + min(abs(y_step3))) * 0.6;
    reliable_indices = find(abs(y_step3) > amp_threshold);
    
    y_final_restored = y_step3; % 默认值，以防无法校正
    if length(reliable_indices) > 50 % 确保有足够多的可靠点
        n_reliable = reliable_indices' - 1; 
        phase_for_regression = unwrapped_phase_all(reliable_indices);
        coeffs = polyfit(n_reliable, phase_for_regression, 2);
        
        n_vec = 0:(L-1);
        phase_correction_curve = coeffs(1)*n_vec.^2 + coeffs(2)*n_vec;
        cfo_correction_factor = exp(-1i * phase_correction_curve);
        y_step4 = y_step3 .* cfo_correction_factor;
        
        on_state_points = y_step4(reliable_indices);
        phase_offset_estimated = mean(angle(on_state_points));
        phase_correction_factor = exp(-1i * phase_offset_estimated);
        y_final_restored = y_step4 .* phase_correction_factor;
    end
    
    % --- 步骤 C: 格式化并保存为 .csv 文件 ---
    % 将最终的复数信号 y_final_restored 分离为 I 和 Q
    iq_data_to_save = [real(y_final_restored); imag(y_final_restored)]; % 格式为 2x1024
    
    % 创建文件名
    filename_csv = fullfile(output_folder, sprintf('restored_ask_sample_%04d.csv', i));
    
    % 使用 writematrix 保存为 .csv 文件
    writematrix(iq_data_to_save, filename_csv);
    
    % 可选: 显示进度
    if mod(i, 10) == 0
        fprintf('已处理并保存 %d / %d 个文件...\n', i, num_files_to_generate);
    end
end

fprintf('批量还原与保存任务完成！\n');
fprintf('%d 个CSV文件已保存在文件夹 "%s" 中。\n', num_files_to_generate, output_folder);