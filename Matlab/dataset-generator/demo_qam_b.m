clc, clear, close all

RndPhase = true;

% (参数部分保持不变)
ModType = 'qam';
M_i = 16;
sps_i = 4;
phaserot = 0;
ini_phase_i = 0;
dfcTs_i = 0;
rolloff = 0.45; 
symbol_order = 'gray';

[Mod_QAM] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

if RndPhase
    Mod_QAM.ini_phase = 2*pi*rand(1,1);
end

L = 1024;
SNR = 24;

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));

delay = 0;
Nsyms = ceil((delay+L+dt+1)/sps_i);
data = randi([0 1],[1 Nsyms*log2(Mod_QAM.M)]);

% --- 1. 生成原始的线性信号 ---
x_linear = qammod_NN(Mod_QAM, data);

% --- 2. 为信号添加非线性失真 ---
% Rapp模型参数
p = 2.0;          
A_sat = 1.0;      

% 峰值幅度归一化
x_peak_power = max(abs(x_linear).^2);
x_normalized_for_pa = x_linear / sqrt(x_peak_power);

% 应用Rapp模型
input_amp = abs(x_normalized_for_pa);
output_amp = input_amp ./ (1 + (input_amp / A_sat).^(2*p)).^(1/(2*p));

% 得到经过非线性失真的信号
x_nonlinear = x_normalized_for_pa .* (output_amp ./ input_amp);

% 功率恢复，以进行公平的SNR比较
power_linear = mean(abs(x_linear).^2);
power_nonlinear_temp = mean(abs(x_nonlinear).^2);
x_nonlinear = x_nonlinear * sqrt(power_linear / power_nonlinear_temp);

fprintf('已为QAM信号添加非线性失真。\n');

% --- 3. 让线性和非线性信号分别通过AWGN信道 ---
y_linear = awgn(x_linear, SNR, 'measured'); 
y_nonlinear = awgn(x_nonlinear, SNR, 'measured'); 

% --- 4. 截取用于分析的突发信号 ---
% 为了在时域上看到完全相同的片段，我们使用相同的延迟和长度
burst_linear = y_linear(delay+dt+(1:L));
burst_nonlinear = y_nonlinear(delay+dt+(1:L));

%% =========================================================================
% <<< --- 修改点：全面的可视化对比部分 --- >>>
% =========================================================================
figure('Name', '非线性失真效果全面对比 (16-QAM)', 'Position', [50, 50, 1600, 800]);

% --- 第一行：星座图对比 ---
subplot(3, 2, 1);
scatter(real(y_linear), imag(y_linear), 25, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
grid on; axis square;
title('1a. 星座图 (线性信号)');
xlabel('I'); ylabel('Q');
lim_max = max(abs([xlim, ylim])); axis([-lim_max, lim_max, -lim_max, lim_max]);

subplot(3, 2, 2);
scatter(real(y_nonlinear), imag(y_nonlinear), 25, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
grid on; axis square;
title('1b. 星座图 (非线性信号)');
xlabel('I'); ylabel('Q');
lim_max = max(abs([xlim, ylim])); axis([-lim_max, lim_max, -lim_max, lim_max]);

% --- 第二行：IQ时域波形对比 ---
subplot(3, 2, 3);
plot(1:L, real(burst_linear), 'b-', 'DisplayName', 'I-Linear');
hold on;
plot(1:L, imag(burst_linear), 'c-', 'DisplayName', 'Q-Linear');
grid on; hold off;
title('2a. IQ波形 (线性信号)');
xlabel('采样点索引'); ylabel('幅度');
legend;
ylim_max = max(abs(ylim)) * 1.1; ylim([-ylim_max, ylim_max]);

subplot(3, 2, 4);
plot(1:L, real(burst_nonlinear), 'r-', 'DisplayName', 'I-Nonlinear');
hold on;
plot(1:L, imag(burst_nonlinear), 'm-', 'DisplayName', 'Q-Nonlinear');
grid on; hold off;
title('2b. IQ波形 (非线性信号)');
xlabel('采样点索引'); ylabel('幅度');
legend;
ylim(subplot(3,2,3), ylim); % 保持Y轴范围一致以便对比

% --- 第三行：功率谱密度(PSD)对比 ---
fs = sps_i; % 假设符号速率为1Hz，则采样率为sps

subplot(3, 2, 5);
pwelch(x_linear, [], [], [], fs, 'centered', 'psd'); % 使用无噪声信号计算PSD更清晰
title('3a. 功率谱密度 (线性信号)');

subplot(3, 2, 6);
pwelch(x_nonlinear, [], [], [], fs, 'centered', 'psd'); % 使用无噪声信号计算PSD更清晰
title('3b. 功率谱密度 (非线性信号)');