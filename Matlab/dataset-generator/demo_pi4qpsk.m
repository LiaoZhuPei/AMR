clc, clear, close all

RndPhase = true;

% --- 1. 设置参数以生成 pi/4-QPSK 信号 ---
ModType = 'pi4qpsk'; % <-- 关键修改: 指定新的调制类型
sps_i = 4;
rolloff = 0.35;
symbol_order = 'gray';

[Mod_QPSK] = Initialize_Modulation_NN(ModType, 'sps',sps_i, 'beta',rolloff, 'symbol_order',symbol_order);

% Random Phase (这将作为全局相位偏移)
if RndPhase
    Mod_QPSK.ini_phase = 2*pi*rand(1,1);
end
L = 1024;
SNR = 36;

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0;
% Number of Symbols
Nsyms = ceil((delay+L+dt+1)/sps_i);

% Generate Random Burst of Information (QPSK 是 2比特/符号)
data = randi([0 1],[1 Nsyms*log2(Mod_QPSK.M)]);

% --- 2. 使用新的 pi4qpskmod_NN 进行调制 ---
x = pi4qpskmod_NN(Mod_QPSK, data); % <-- 关键修改: 调用新函数

% channel
y_noisy = awgn(x, SNR, 'measured'); % 注意这里对 scaled signal 加噪声
y = y_noisy(delay+dt+(1:L));

%% Visualization Section (与您提供的完全一致)
% 1. 绘制IQ波形图
figure;
plot(1:L, real(y), 'b-', 'LineWidth', 1);
hold on;
plot(1:L, imag(y), 'r--', 'LineWidth', 1);
grid on;
title('pi/4-QPSK IQ 波形图 (IQ Waveform)');
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

% 2. 绘制星座图
figure;
scatter(real(y), imag(y), 36, 'b', 'filled');
grid on;
axis square;
title('pi/4-QPSK 星座图 (Constellation Diagram)');
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
lim_max = max(abs([xlim, ylim]));
axis([-lim_max, lim_max, -lim_max, lim_max]);
% 叠加理想星座点以供参考
hold on;
ideal_points1 = exp(1i * Mod_QPSK.Constellation1);
ideal_points2 = exp(1i * Mod_QPSK.Constellation2);
plot(ideal_points1, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(ideal_points2, 'kx', 'MarkerSize', 10, 'LineWidth', 2);
legend('接收信号点', '星座图1 (标准)', '星座图2 (旋转)');
hold off;


% 3. 绘制功率谱密度 (PSD)
fs = sps_i; 
figure;
pwelch(y, [], [], [], fs, 'centered', 'psd');
title('pi/4-QPSK 功率谱密度 (Power Spectral Density)');

% 1. 定义要保存的文件名
%    您可以根据需要修改文件名，例如区分QPSK和DQPSK
output_filename = 'My_Pi4QPSK_Sample_36dB.mat';

% 2. 将复数信号 y 转换为 2x1024 的 I/Q 格式
%    这与您主仿真程序中的格式完全一致
iq_data = [real(y); imag(y)]; % <--- 关键：创建名为 iq_data 的变量

% 3. 保存变量到 .mat 文件
%    我们将只保存 iq_data 这个变量到文件中
save(output_filename, 'iq_data');

fprintf('\n信号已成功生成并保存到文件: %s\n', output_filename);
fprintf('文件中的变量名为: "iq_data"\n');