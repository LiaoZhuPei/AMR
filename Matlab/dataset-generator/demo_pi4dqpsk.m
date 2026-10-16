clc, clear, close all

RndPhase = true;

% --- 1. 设置参数以生成 pi/4-DQPSK 信号 ---
ModType = 'pi4dqpsk'; % <-- 关键修改1
sps_i = 4;
ini_phase_i = 0;
dfcTs_i = 0;
rolloff = 0.35;
symbol_order = 'gray';

[Mod_DQPSK] = Initialize_Modulation_NN(ModType, 'sps',sps_i, 'beta',rolloff, 'symbol_order',symbol_order);

% Random Phase (这将作为全局相位偏移)
if RndPhase
    Mod_DQPSK.ini_phase = 2*pi*rand(1,1);
end
L = 1024;
SNR = 36;

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0;
% Number of Symbols
Nsyms = ceil((delay+L+dt+1)/sps_i);

% Generate Random Burst of Information (DQPSK 是 2比特/符号)
data = randi([0 1],[1 Nsyms*log2(Mod_DQPSK.M)]);

% --- 2. 使用新的 pi4dqpskmod_NN 进行调制 ---
x = pi4dqpskmod_NN(Mod_DQPSK, data); % <-- 关键修改2

% ==================== temp修改点 ====================
% 1. 定义一个缩放因子/增益
%gain = 0.4;
        
% 2. 将原始信号 x 乘以该增益，得到缩放后的信号 x_scaled
%x_scaled = x * gain;
% 现在 x_scaled 的I分量幅度就在 ±0.3 左右了

% channel
%y = awgn(x_scaled,SNR);

% ... (channel, Get Burst等部分保持不变) ...
y = awgn(x,SNR, 'measured');
y = y(delay+dt+(1:L));

%% Visualization Section
% 1. 绘制IQ波形图
figure; % 创建一个新的图形窗口
plot(1:L, real(y), 'b-', 'LineWidth', 1); % 绘制I分量
hold on; % 保持当前图形，以便在上面叠加新的曲线
plot(1:L, imag(y), 'r--', 'LineWidth', 1); % 绘制Q分量
grid on;
title('IQ 波形图 (IQ Waveform)');
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

% 2. 绘制星座图
% 使用 scatter 函数可以轻松绘制
figure;
scatter(real(y), imag(y), 36, 'b', 'filled'); % 'b'代表蓝色, 'filled'填充点
grid on;
axis square; % 使坐标轴比例相同，星座图更直观
title('星座图 (Constellation Diagram)');
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
% 设定坐标轴范围，使其围绕原点对称，看得更清楚
lim_max = max(abs([xlim, ylim]));
axis([-lim_max, lim_max, -lim_max, lim_max]);

% 3. 绘制功率谱密度 (PSD)
% 在仿真中，通常假设符号速率为1Hz，则采样率 fs 等于每个符号的采样点数 sps_i
fs = sps_i; 

figure;
% 使用 pwelch 函数来估计PSD，它比简单的FFT更平滑、准确
pwelch(y, [], [], [], fs, 'centered', 'psd');
title('功率谱密度 (Power Spectral Density)');
% pwelch函数会自动处理标签，但如果想自定义，可以如下操作：
% [Pxx, F] = pwelch(y, hann(256), 128, 1024, fs, 'centered', 'psd');
% plot(F, 10*log10(Pxx));
% grid on;
% title('功率谱密度 (Power Spectral Density)');
% xlabel('归一化频率 (Hz)');
% ylabel('功率/频率 (dB/Hz)');

% 1. 定义要保存的文件名
%    您可以根据需要修改文件名，例如区分QPSK和DQPSK
output_filename = 'My_Pi4DQPSK_Sample_36dB.mat';

% 2. 将复数信号 y 转换为 2x1024 的 I/Q 格式
%    这与您主仿真程序中的格式完全一致
iq_data = [real(y); imag(y)]; % <--- 关键：创建名为 iq_data 的变量

% 3. 保存变量到 .mat 文件
%    我们将只保存 iq_data 这个变量到文件中
save(output_filename, 'iq_data');

fprintf('\n信号已成功生成并保存到文件: %s\n', output_filename);
fprintf('文件中的变量名为: "iq_data"\n');