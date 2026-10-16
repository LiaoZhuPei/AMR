% =========================================================================
%                reverse_signal_distortions.m (最终增强版)
% =========================================================================
clc, clear, close all

% --- 首先，我们先运行demo_ask.m来生成一个带失真的信号 ---
% (这里直接复制了demo_ask.m的核心生成逻辑，以便脚本能独立运行)
SimulateDistortion = true; 
amplitude_on = 1.0; amplitude_off = 0.36; 
total_rotations_in_frame = 0.6;
initial_phase_offset_rad = pi/4; 
dc_offset_i = -0.5; dc_offset_q = -0.5;
global_gain = 3000; SNR = 25; L = 1024; sps_i = 4;
ModType = 'ask'; M_i = 2;
[Mod_ASK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i);
filter_delay = Mod_ASK.PulseShaping.delay;
num_required_samples = L + sps_i;
Nsyms = ceil(num_required_samples / sps_i) + ceil(filter_delay * 2 / sps_i);
data = randi([0 1],[1 Nsyms*log2(M_i)]);
x_raw_filtered = askmod_NN(Mod_ASK, data); % 假设askmod_NN是平滑版本
map_on = (x_raw_filtered == 1); map_off = (x_raw_filtered == -1);
x_imperfect_ook = zeros(size(x_raw_filtered));
x_imperfect_ook(map_on) = amplitude_on; x_imperfect_ook(map_off) = amplitude_off;
x_distorted = x_imperfect_ook;
if SimulateDistortion
    normalized_frequency = total_rotations_in_frame / L;
    n = 0:(length(x_imperfect_ook)-1);
    cfo_rotation_factor = exp(1i * (2 * pi * normalized_frequency * n + initial_phase_offset_rad));
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

% 我们要还原的信号
y_distorted_signal = y_final_burst;


%% ======================= 逆向还原过程 (最终增强版) =======================

% --- 步骤 1-3: 与之前版本完全相同 ---
y_step1 = conj(y_distorted_signal);
rms_val = sqrt(mean(abs(y_step1).^2));
y_step2 = y_step1 / rms_val;
dc_offset_estimated = mean(y_step2);
y_step3 = y_step2 - dc_offset_estimated;

% --- 步骤 4: CFO校正 (与之前版本完全相同) ---
amp_threshold_cfo = (max(abs(y_step3)) + min(abs(y_step3))) * 0.6;
reliable_indices = find(abs(y_step3) > amp_threshold_cfo);
phase_reliable = angle(y_step3(reliable_indices));
unwrapped_phase_reliable = unwrap(phase_reliable);
n_reliable = reliable_indices' - 1;
coeffs = polyfit(n_reliable, unwrapped_phase_reliable, 1);
omega_estimated = coeffs(1);
f_offset_estimated_normalized = omega_estimated / (2*pi);
fprintf('估计的归一化频偏为: %f\n', f_offset_estimated_normalized);
n_vec = 0:(L-1);
cfo_correction_factor = exp(-1i * 2 * pi * f_offset_estimated_normalized * n_vec);
y_step4 = y_step3 .* cfo_correction_factor;

% --- 步骤 5: 相位校正 (与之前版本完全相同) ---
on_state_points = y_step4(reliable_indices);
phase_offset_estimated = mean(angle(on_state_points));
phase_correction_factor = exp(-1i * phase_offset_estimated);
y_step5_phase_corrected = y_step4 .* phase_correction_factor;

% --- 【关键新增】：步骤 6: 幅度恢复/均衡 (硬限幅) ---
fprintf('正在进行幅度恢复...\n');

% 提取校正后的I通道信号
i_channel_corrected = real(y_step5_phase_corrected);

% 估计"ON"和"OFF"状态的理想电平
% 我们可以通过找到幅度直方图的两个峰值来做到这一点，
% 但一个更简单的方法是直接计算ON/OFF点的平均值。
amp_threshold_decision = 0.5; % 假设归一化后0.5是ON/OFF的决策边界
on_indices = find(i_channel_corrected > amp_threshold_decision);
off_indices = find(i_channel_corrected <= amp_threshold_decision);

% 创建一个新的、理想化的I通道
i_channel_restored = zeros(size(i_channel_corrected));
i_channel_restored(on_indices) = 1.0;  % 所有ON点强制为1.0
i_channel_restored(off_indices) = 0.36; % 所有OFF点强制为我们已知的理论值

% Q通道被强制为零
q_channel_restored = zeros(size(i_channel_corrected));

% 组合成最终的、高度理想化的信号
y_final_restored = i_channel_restored + 1i * q_channel_restored;


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

% 原始失真信号
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title('1. 原始失真信号');
grid on; legend('I', 'Q');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

% 最终还原信号
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 最终还原的高度理想化信号 (带幅度恢复)');
grid on; legend('I', 'Q'); ylim([-0.5 1.5]);

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('还原后信号星座图 (带幅度恢复)');
grid on; axis equal; xlim([-0.5 1.5]); ylim([-0.5 0.5]);