clc, clear, close all

RndPhase = true;

% --- 1. 设置参数以生成 GMSK 信号 ---
ModType = 'gmsk';       % <-- 关键修改1: 调制类型
sps_i = 4;
BT_product = 0.3;       % <-- GMSK 特有参数: 带宽时间乘积 (常用值为0.3)

% 使用更新后的 Initialize_Modulation_NN 函数
[Mod_GMSK] = Initialize_Modulation_NN(ModType, 'sps', sps_i, 'BT', BT_product);

% Random Phase (这将作为全局相位偏移)
if RndPhase
    Mod_GMSK.ini_phase = 2*pi*rand(1,1);
end

% --- 仿真参数 (与您的模板保持一致) ---
L = 1024;
SNR = 36; % 使用高信噪比以便清晰观察信号特征

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0; % 注意: 在我们新的 gmskmod_NN 中，滤波器延迟在函数内部处理

% Number of Symbols (GMSK M=2, 所以是比特数)
Nsyms = ceil((delay+L+dt+1)/sps_i);

% Generate Random Burst of Information (GMSK 是 1比特/符号)
data = randi([0 1],[1 Nsyms*log2(Mod_GMSK.M)]);

% --- 2. 使用新的 gmskmod_NN 进行调制 ---
x = gmskmod_NN(Mod_GMSK, data); % <-- 关键修改2

% --- 信号缩放和信道 ---
% GMSK是恒包络信号, 其原始幅度为1。为了与您的模板输出范围保持一致，
% 我们可以同样应用一个增益因子。
gain = 0.4;
x_scaled = x * gain;

% 添加高斯白噪声
y_noisy = awgn(x_scaled, SNR, 'measured'); % 使用'measured'更准确

% 截取随机起始位置的片段 (与您的模板逻辑一致)
% 注意：这里的 delay=0，因为 gmskmod_NN 内部已处理延迟
y = y_noisy(delay+dt+(1:L));

%% ==================== 可视化部分 ====================
% (此部分与您的模板完全相同，但我们会解释GMSK的特有现象)

% 1. 绘制IQ波形图
figure;
plot(1:L, real(y), 'b-', 'LineWidth', 1);
hold on;
plot(1:L, imag(y), 'r--', 'LineWidth', 1);
grid on;
title('GMSK IQ 波形图 (IQ Waveform)');
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

% 2. 绘制星座图
figure;
scatter(real(y), imag(y), 10, 'b', 'filled'); % 使用更小的点以看清轨迹
grid on;
axis square;
title('GMSK 星座图 (Constellation Diagram)');
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
lim_max = max(abs([xlim, ylim])) * 1.1; % 稍微放大坐标轴
axis([-lim_max, lim_max, -lim_max, lim_max]);
% --- GMSK 特征说明 ---
% GMSK是恒包络、连续相位的调制。因此，它的星座图不是像QPSK那样的几个
% 离散点，而是一个**圆环**。所有信号点都以恒定的幅度在圆上平滑移动，
% 永远不会跳变。您看到的将是一个被噪声加宽的明亮圆环。

% 3. 绘制功率谱密度 (PSD)
fs = sps_i; 

figure;
pwelch(y, [], [], [], fs, 'centered', 'psd');
title('GMSK 功率谱密度 (Power Spectral Density)');
% --- GMSK 特征说明 ---
% GMSK的主要优点体现在频谱上。由于高斯预滤波，它的频谱非常紧凑，
% 主瓣窄，旁瓣衰减很快。这使得它具有很高的频谱效率和良好的邻道干扰抑制能力。