% =========================================================================
%                reverse_signal_distortions.m (最终正确版 V2)
% =========================================================================
clc, clear, close all

% --- 首先，我们先运行demo_ask.m来生成一个带失真的信号 ---
% (这部分代码与之前完全相同)
SimulateDistortion = true; 
amplitude_on = 1.0; amplitude_off = 0.36; 
total_rotations_in_frame = 0.6;
initial_phase_offset_rad = rand() * 2 * pi;
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
    % 为了更好地模拟弧形，我们可以在这里也加入一个轻微的二次相位
    n = 0:(length(x_imperfect_ook)-1);
    quadratic_phase = 0.000001 * n.^2; % 一个微小的频率漂移项
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

% 我们要还原的信号
y_distorted_signal = y_final_burst;


%% ======================= 逆向还原过程 (最终正确版 V2) =======================

% --- 步骤 1-3: 预处理 (与之前版本完全相同) ---
y_step1 = conj(y_distorted_signal);
rms_val = sqrt(mean(abs(y_step1).^2));
y_step2 = y_step1 / rms_val;
dc_offset_estimated = mean(y_step2);
y_step3 = y_step2 - dc_offset_estimated;

% --- 步骤 4: 校正载波频率偏移和频率漂移 - 【最终正确算法】 ---

% 先对所有点的相位进行解缠绕
phase_all = angle(y_step3);
unwrapped_phase_all = unwrap(phase_all);
% 找出可靠点的索引
amp_threshold = (max(abs(y_step3)) + min(abs(y_step3))) * 0.6;
reliable_indices = find(abs(y_step3) > amp_threshold);
% 只用可靠点进行回归
n_reliable = reliable_indices' - 1; 
phase_for_regression = unwrapped_phase_all(reliable_indices);

% 【关键修正】：使用 polyfit 拟合一个二阶多项式 (二次曲线)
coeffs = polyfit(n_reliable, phase_for_regression, 2);
p2 = coeffs(1); % 二次项系数 (与频率漂移相关)
p1 = coeffs(2); % 一次项系数 (与频率偏移相关)
p0 = coeffs(3); % 常数项 (与初始相位相关)
fprintf('估计的相位模型: %.2e*n^2 + %.4f*n + %.4f\n', p2, p1, p0);

% 创建一个反向旋转的、包含二次和一次项的校正相位
n_vec = 0:(L-1);
phase_correction_curve = p2*n_vec.^2 + p1*n_vec;
cfo_correction_factor = exp(-1i * phase_correction_curve);
y_step4 = y_step3 .* cfo_correction_factor;

% --- 步骤 5: 校正剩余的、由p0代表的常数相位偏移 ---
% 在频率校正后，我们只需校正剩余的常数相位
% 我们可以直接使用拟合出的p0，或者用更稳健的平均法
on_state_points = y_step4(reliable_indices);
phase_offset_estimated = mean(angle(on_state_points));
phase_correction_factor = exp(-1i * phase_offset_estimated);
y_final_restored = y_step4 .* phase_correction_factor;


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);
% ... (可视化部分代码与之前完全相同) ...
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title('1. 原始失真信号');
grid on; legend('I', 'Q');
subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 最终还原的近似理想信号 (最终正确版)');
grid on; legend('I', 'Q'); ylim([-1.5 1.5]);
subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('还原后信号星座图 (最终正确版)');
grid on; axis equal; xlim([-0.5 1.5]);