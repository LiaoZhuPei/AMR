clc, clear, close all

%随机初始相位
RndPhase = true;

%% PSK
% M: Modulation order (2 (default), 4, or 8) ---> M=2,即为BPSK
% sps: samples per symbol (default: 8) ---> 每个符号的采样点数
% ini_phase: Carrier initial phase (default: 0) ---> 载波的默认初始相位，PndPhase=false时使用
% phase_cont: Phase continuity ('discont' | 'cont' (default))
% dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
% symbol_order: Symbol order ('gray' (default) | 'bin')
ModType = 'oqpsk';
M_i = 4;
sps_i = 4;
phaserot = 0;           %施加于星座图的额外相位旋转
ini_phase_i = 0;
dfcTs_i = 0;            %归一化偏频，用于FSK调制或模拟接收机中的频率偏移
rolloff = 0.45;         %滚降系数，用于脉冲成形滤波器（如升余弦滤波器）的参数
symbol_order = 'gray';  %符号的编码顺序，采用“格雷码”（Gray code）

% M: Modulation order (2 (default), 4, 8, 16)
% phaserot: phase rotation of the modulation in radians (default: 0)
% sps: samples per symbol (default: 8)
% ini_phase: Carrier initial phase (default: 0)
% dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
% beta: Roll-off factor for root raised cosine pulse (default: 0.35)
% symbol_order: Symbol order ('gray' (default) | 'bin')
[Mod_OQPSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

% Random Phase
if RndPhase
    Mod_OQPSK.ini_phase = 2*pi*rand(1,1);
end

% Randomly Select Burst Prameters
L = 1024;
SNR = 36;

%%生成随时间偏移
% 引入了时间不确定性，模拟采样定时误差
dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0;

%%生成数据并进行调制
% Number of Symbols
% 计算需要生成多少个调制符号才能满足后续的截取需求
Nsyms = ceil((delay+L+dt+1)/sps_i);

% Generate Random Burst of Information
%生成随机的二进制数据比特流
%log2(Mod_PSK.M)计算每个符号承载多少个比特（对于BPSK, log2(2) = 1）
%Nsyms * log2(...): 计算所需的总比特数
data = randi([0 1],[1 Nsyms*log2(Mod_OQPSK.M)]);

x = oqpskmod_NN(Mod_OQPSK, data);

%% channel
% 3. 将缩放后的信号 x_scaled 送入AWGN信道
% awgn函数会先测量 x_scaled 的功率，然后加入正确比例的噪声
y = awgn(x, SNR);
% ======================================================

%% channel
%y = awgn(x,SNR);

%% Get Burst
%截取最终的突发信号
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











