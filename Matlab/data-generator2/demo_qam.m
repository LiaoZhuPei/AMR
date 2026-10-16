clc, clear, close all

RndPhase = true;

% M: Modulation order (8, 16 (default), 32, 64, 128)
% phaserot: phase rotation of the modulation in radians (default: 0)
% sps: samples per symbol (default: 8)
% ini_phase: Carrier initial phase (default: 0)
% dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
% beta: Roll-off factor for root raised cosine pulse (default: 0.35)
% symbol_order: Symbol order ('gray' (default) | 'bin')
ModType = 'qam';
M_i = 16;
sps_i = 4;
phaserot = 0;
ini_phase_i = 0;
dfcTs_i = 0;
rolloff = 0.35;
symbol_order = 'gray';

[Mod_QAM] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

% Random Phase
if RndPhase
    Mod_QAM.ini_phase = 2*pi*rand(1,1);
end

% Randomly Select Burst Prameters
L = 500;
SNR = 24;

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));

delay = 0;
% Number of Symbols
Nsyms = ceil((delay+L+dt+1)/sps_i);

% Generate Random Burst of Information
data = randi([0 1],[1 Nsyms*log2(Mod_QAM.M)]);

x = qammod_NN(Mod_QAM, data);

%signal_power = mean(abs(x).^2);
%x_normalized = x / sqrt(signal_power); % 使用幅度的平方根来缩放

% --- 将归一化后的信号传入 awgn ---
y = awgn(x, SNR, 'measured'); % 现在awgn在单位功率信号上工作
y2 = y;
% Get Burst
y = y(delay+dt+(1:L));

%% Visualization Section
% 1. 绘制IQ波形图
figure; % 创建一个新的图形窗口
plot(1:L, real(y), 'b-', 'LineWidth', 1); % 绘制I分量
hold on; % 保持当前图形，以便在上面叠加新的曲线
plot(1:L, imag(y), 'r-', 'LineWidth', 1); % 绘制Q分量
grid on;
title('IQ 波形图 (IQ Waveform)');
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

figure('Name', 'Correct Constellation Comparison');

% 首先，下采样接收信号 y，只获取每个符号的判决采样点
received_symbols = downsample(y, sps_i, mod(dt, sps_i));

% 绘制这些接收到的判决点 (蓝色散点)
scatter(real(received_symbols), imag(received_symbols), 36, 'b', 'filled', 'DisplayName', sprintf('接收符号判决点 (SNR=%d dB)', SNR), 'MarkerFaceAlpha', 0.8);
hold on;

grid on;
axis square;
title(sprintf('APSK-%d 星座图对比 (正确方法)', M_i));
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
legend('show', 'Location', 'best');
hold off;
lim_max = max(abs([xlim, ylim])) * 1.1;
if lim_max < 1.5, lim_max = 1.5; end
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











