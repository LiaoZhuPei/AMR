clc, clear, close all

RndPhase = true;

%% FSK
% ... (这部分代码保持不变) ...
ModType = 'fsk';
M_i = 2;
sps_i = 4;
h_i = 1;
ini_phase_i = 0;
phase_cont = 'cont';
dfcTs_i = 0;
symbol_order = 'gray';
[Mod_FSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'h',h_i,'ini_phase',ini_phase_i,'phase_cont',phase_cont,'dfcTs',dfcTs_i,'symbol_order',symbol_order);

% ... (这部分代码保持不变) ...
if RndPhase
    Mod_FSK.ini_phase = 2*pi*rand(1,1);
end

L = 1024;
SNR =18;
dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0;
Nsyms = ceil((delay+L+dt+1)/sps_i);
data = randi([0 1],[1 Nsyms*log2(Mod_FSK.M)]);
x = fskmod_NN(Mod_FSK, data);
y_noisy = awgn(x,SNR); % 将带噪声的信号存储在临时变量中
y = y_noisy(delay+dt+(1:L)); % 截取burst

%% Visualization Section
% 1. 绘制IQ波形图
figure; % 创建一个新的图形窗口
plot(1:L, real(y), 'b-', 'LineWidth', 1); % 绘制I分量
hold on; % 保持当前图形，以便在上面叠加新的曲线
plot(1:L, imag(y), 'r-', 'LineWidth', 1); % 绘制Q分量
grid on;
title('归一化后的IQ波形图 (Normalized IQ Waveform)'); % 更新标题
xlabel('采样点索引 (Sample Index)');
ylabel('归一化幅度 (Normalized Amplitude)'); % 更新Y轴标签
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

% 2. 绘制星座图
% ... (星座图绘图代码保持不变，但由于y已归一化，它的范围也会变化) ...
figure;
scatter(real(y), imag(y), 36, 'b', 'filled');
grid on;
axis square;
title('归一化后的星座图 (Normalized Constellation Diagram)'); % 更新标题
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
lim_max = max(abs([xlim, ylim]));
axis([-lim_max, lim_max, -lim_max, lim_max]);
% 由于FSK是恒包络的，归一化后星座图会变成一个半径小于等于1的圆环

% 3. 绘制功率谱密度 (PSD)
% ... (PSD绘图代码保持不变) ...
fs = sps_i; 
figure;
pwelch(y, [], [], [], fs, 'centered', 'psd');
title('归一化后的功率谱密度 (Normalized Power Spectral Density)'); % 更新标题







