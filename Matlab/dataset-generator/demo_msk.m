clc, clear, close all

RndPhase = true;

%% MSK
ModType = 'msk';
%M_i = 2;
sps_i = 4;
%h_i = 1;
ini_phase_i = 0;
%phase_cont = 1;
dfcTs_i = 0;
%symbol_order = 'gray';
[Mod_MSK] = Initialize_Modulation_NN(ModType,'sps',sps_i,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i);

% Random Phase
if RndPhase
    Mod_MSK.ini_phase = 2*pi*rand(1,1);
end

% Randomly Select Burst Prameters
L = 1024;
SNR = 20;

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));

delay = 0;
% Number of Symbols
Nsyms = ceil((delay+L+dt+1)/sps_i);

% Generate Random Burst of Information
data = randi([0 1],[1 Nsyms*log2(Mod_MSK.M)]);

x = fskmod_NN(Mod_MSK, data);

% channel
y = awgn(x,SNR);

% Get Burst
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