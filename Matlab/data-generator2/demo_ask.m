% --- ASK调制的主调用脚本 (类似代码1) ---
clc, clear, close all

RndPhase = true; % 是否使用随机初始相位

% --- ASK调制器参数 ---
% M: 调制阶数 (e.g., 2, 4, 8 (default), 16)
% phaserot: 调制的相位旋转 (弧度, 默认: 0)
% sps: 每个符号的采样点数 (默认: 8)
% ini_phase: 载波初始相位 (默认: 0)
% dfcTs: 载波频偏与符号周期的乘积 (默认: 0)
% beta: 根升余弦脉冲的滚降系数 (默认: 0.35)
% symbol_order: 符号顺序 ('gray' (default) | 'bin')
ModType = 'ask';
M_i = 2;         % ASK调制阶数，例如8-ASK
sps_i = 4;       % 每个符号的采样点数
phaserot = 0;    % 相位旋转
ini_phase_i = 0; % 初始相位
dfcTs_i = 0;     % 归一化频偏
rolloff = 0.35;  % 滚降系数
symbol_order = 'gray'; % 采用格雷码

% 使用与代码2中类似的初始化函数来创建调制器对象
% 假设 Initialize_Modulation_NN 函数已根据代码2的逻辑实现
[Mod_ASK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

% 设置随机相位
if RndPhase
    Mod_ASK.ini_phase = 2*pi*rand(1,1);
end

% --- 随机选择突发信号参数 ---
L = 1024;  % 突发长度
SNR = 18;  % 信噪比 (dB)

% 随机采样时间偏移
dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));

delay = 0; % 延迟 (此处为0)

% 计算所需的符号数
Nsyms = ceil((delay+L+dt+1)/sps_i);

% 生成随机的二进制信息比特流
data = randi([0 1],[1 Nsyms*log2(Mod_ASK.M)]);

% --- 调用ASK调制函数 ---
x = askmod_NN(Mod_ASK, data);

% --- 通过信道 ---
y = awgn(x,SNR);

% --- 获取突发信号 ---
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