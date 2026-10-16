%% --- APSK调制与可视化完整脚本 ---
clc; 
clear; 
close all;

%% 1. 初始化和参数设置
% --- 仿真控制参数 ---
RndPhase = true; % 是否使用随机初始相位

% --- APSK调制器参数 ---
ModType = 'apsk';       % 调制类型
M_i = 16;               % APSK调制阶数 (16或32)
sps_i = 4;              % 每个符号的采样点数 (Samples Per Symbol)
phaserot = 0;           % 整体星座图的相位旋转 (弧度)
ini_phase_i = 0;        % 载波初始相位 (弧度)
dfcTs_i = 0;            % 归一化载波频率偏移
rolloff = 0.35;         % 根升余弦滤波器的滚降系数
symbol_order = 'bin';   % 符号映射方式 ('bin' 或 'gray')

% --- 信道和信号参数 ---
L = 1024;  % 突发信号长度 (采样点数)，增加长度以获得更清晰的星座图
SNR = 50;  % 信噪比 (dB)，可以调整此值观察噪声影响

%% 2. 调制器和信号生成
% --- 创建调制器对象 ---
% 假设 Initialize_Modulation_NN 函数已准备好
try
    [Mod_APSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
catch ME
    error('请确保 Initialize_Modulation_NN 函数存在且支持APSK类型。\n错误信息: %s', ME.message);
end

% --- 设置随机效应 ---
if RndPhase
    Mod_APSK.ini_phase = 2*pi*rand(1,1); % 随机初始相位
end

% 随机采样时间偏移，模拟定时误差
dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));

delay = 0; % 信号延迟 (此处为0)

% --- 生成随机数据和理想信号 ---
% 计算所需的符号总数以确保信号长度足够
Nsyms = ceil((delay+L+dt+1)/sps_i);

% 生成随机的二进制信息比特流
data = randi([0 1],[1 Nsyms*log2(Mod_APSK.M)]);

% 调用APSK调制函数生成理想的、无噪声的基带信号 x
fprintf('正在生成理想APSK信号...\n');
x = apskmod_NN(Mod_APSK, data);

%% 3. 信道模拟
% --- 在添加噪声前，将信号功率归一化为 1 ---
signal_power = mean(abs(x).^2);
x_normalized = x / sqrt(signal_power); % 使用幅度的平方根来缩放

% (可选) 验证归一化是否成功
fprintf('归一化后的信号功率是: %f\n', mean(abs(x_normalized).^2)); % 这应该是1.0

fprintf('正在添加高斯白噪声 (SNR = %d dB)...\n', SNR);
% --- 将归一化后的信号传入 awgn ---
y_noisy = awgn(x_normalized, SNR, 'measured'); % 现在awgn在单位功率信号上工作

% --- 截取最终的接收信号 y 和对应的理想信号段 ---
% 注意：后续的理想信号也要用归一化后的版本进行比较
x_segment = x_normalized(delay+dt+(1:L)); % 使用 x_normalized
y = y_noisy(delay+dt+(1:L));

% --- 计算并验证实际的信噪比 ---
% 噪声 = 接收信号 - 理想信号
noise_added = y - x_segment; 
noise_power = mean(abs(noise_added).^2);
segment_signal_power = mean(abs(x_segment).^2);

% 根据测量到的功率，手动计算真实的信噪比
actual_snr_db = 10 * log10(segment_signal_power / noise_power);

fprintf('截取段的信号功率是: %f\n', segment_signal_power);
fprintf('添加的噪声的实际功率是: %f\n', noise_power);
fprintf('根据功率计算出的实际信噪比是: %.2f dB\n', actual_snr_db);
fprintf('================ 诊断信息结束 ================\n');

%% 4. 可视化分析 (Visualization Section)
fprintf('正在绘制结果图...\n');

% --- 图1: IQ 波形图 ---
figure('Name', 'IQ Waveform');
plot(1:L, real(y), 'b-', 'LineWidth', 1); % 绘制I分量
hold on;
plot(1:L, imag(y), 'r-', 'LineWidth', 1); % 绘制Q分量
grid on;
title(sprintf('APSK-%d IQ 波形图 (SNR = %d dB)', M_i, SNR));
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

% --- 图2: 星座图对比 (正确的可视化方法) ---
figure('Name', 'Correct Constellation Comparison');

% 首先，下采样接收信号 y，只获取每个符号的判决采样点
received_symbols = downsample(y, sps_i, mod(dt, sps_i));

% 绘制这些接收到的判决点 (蓝色散点)
scatter(real(received_symbols), imag(received_symbols), 36, 'b', 'filled', 'DisplayName', sprintf('接收符号判决点 (SNR=%d dB)', SNR), 'MarkerFaceAlpha', 0.8);
hold on;

% 然后，直接生成并绘制理想的、完美的星座图作为参考 (红色圆圈)
ideal_constellation = apsk_constellation_NN(M_i);
scatter(real(ideal_constellation), imag(ideal_constellation), 150, 'r', 'o', 'LineWidth', 2, 'DisplayName', '理想星座点');

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



% --- 图3: 功率谱密度 (PSD) ---
figure('Name', 'Power Spectral Density');

% 采样率 fs 等于每个符号的采样点数 sps_i (当符号速率归一化为1时)
fs = sps_i; 

% 使用 pwelch 函数来估计PSD，结果更平滑
pwelch(y, hann(256), 128, 1024, fs, 'centered', 'psd');
title(sprintf('APSK-%d 功率谱密度 (滚降系数 β = %.2f)', M_i, rolloff));

fprintf('所有操作完成！\n');