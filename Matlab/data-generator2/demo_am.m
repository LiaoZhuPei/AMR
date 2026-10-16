% 主 AM 仿真脚本

clc, clear, close all

%% 1. 仿真参数设置
SNR = 20;               % 信噪比 (dB)
RndPhase = true;        % 是否使用随机初始相位

% --- AM 调制器参数 ---
ModType = 'am';
m_i = 0.8;              % 调制指数 (0.8 表示 80% 调制，< 1)
sps_i = 4;              % 上采样因子
ini_phase_i = 0;        % 默认初始相位 (如果 RndPhase 为 true 会被覆盖)

% --- 基带和载波频率参数 ---
fs_baseband = 6e3;      % 基带信号的采样率 (Hz), 例如 6 kHz
fc = 8e3;              % 载波频率 (Hz), 例如 24 kHz
fs_rf = fs_baseband * sps_i; % 调制后信号的采样率 (48 kHz)

% 计算归一化的载波频率 (fc / fs_rf)，这是函数需要的参数
dfc_div_fs_i = fc / fs_rf; 

% 初始化AM调制器结构体
[Mod_AM] = Initialize_Modulation_NN(ModType, 'sps', sps_i, ...
    'ini_phase', ini_phase_i, 'dfc_div_fs', dfc_div_fs_i, 'm', m_i);

% 如果设置了随机相位，则覆盖默认值
if RndPhase
    Mod_AM.ini_phase = 2 * pi * rand(1, 1);
end

%% 2. 生成基带消息信号
% 创建一个由两个正弦波组成的基带信号，以便在频谱中清晰地看到边带
L_baseband = 1024;      % 基带信号长度
T_baseband = 1 / fs_baseband;
t_baseband = (0:L_baseband-1) * T_baseband;

fm1 = 500;  % 第一个频率成分 (500 Hz)
fm2 = 1200; % 第二个频率成分 (1200 Hz)
m_t = 0.7*sin(2*pi*fm1*t_baseband) + 0.3*cos(2*pi*fm2*t_baseband);

%% 3. 调用AM调制器并模拟信道
% 调用调制函数生成理想的AM信号
x = ammod_NN(Mod_AM, m_t);

% 通过AWGN信道添加噪声
y = awgn(x, SNR, 'measured');

%% 4. 可视化部分

L_rf = length(y); % 获取接收信号的长度

% --- 图 1: 完整的 AM IQ 时域波形与包络 ---
figure('Name', 'AM IQ 时域波形');

L_rf = length(y);
% --- 使用采样点索引作为X轴，以获得更清晰的视图 ---
sample_index = 1:L_rf;
if L_rf > 1024 % 确保长度与神经网络输入一致
    y = y(1:1024);
    sample_index = 1:1024;
    L_rf = 1024;
end

% 绘制 I 分量 (实部)
plot(sample_index, real(y), 'b-', 'DisplayName', 'I 分量 (实部)'); 
hold on;

% *** 新增：绘制 Q 分量 (虚部) ***
plot(sample_index, imag(y), 'g-', 'DisplayName', 'Q 分量 (虚部)');

% 准备用于绘图的理想包络线和基带信号
ideal_envelope = 1 + Mod_AM.m * resample(m_t/max(abs(m_t)), sps_i, 1);
if length(ideal_envelope) > L_rf
    ideal_envelope = ideal_envelope(1:L_rf);
end

% 绘制理论上的包络线 (无噪声)
plot(sample_index, ideal_envelope, 'r--', 'LineWidth', 1.5, 'DisplayName', '理论上包络');
plot(sample_index, -ideal_envelope, 'r--', 'LineWidth', 1.5, 'DisplayName', '理论下包络');

grid on;
title(sprintf('AM IQ 时域波形 (长度: %d点, SNR=%d dB, m=%.1f)', L_rf, SNR, Mod_AM.m));
xlabel('采样点索引 (Sample Index)');
ylabel('幅度');
legend('show', 'Location', 'best');
xlim([1, 500]); % 放大显示前面一部分波形细节
hold off;


% --- 图 2: 星座图轨迹 ---
figure('Name', 'AM 星座图');
% 对于AM，信号点应在I轴上的一条线段上移动
scatter(real(y), imag(y), 10, 'filled', 'MarkerFaceAlpha', 0.2);

grid on;
axis square; % 保持坐标轴比例一致
title('AM 星座图轨迹');
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
lim_max = max(abs([xlim, ylim])) * 1.1;
if lim_max < 1.5, lim_max = 1.5; end
axis([-lim_max, lim_max, -lim_max, lim_max]);
hold on;
% 绘制理论上的轨迹线（I轴上的一段）
theory_I_min = 1 - Mod_AM.m;
theory_I_max = 1 + Mod_AM.m;
plot([theory_I_min, theory_I_max], [0, 0], 'r-', 'LineWidth', 3, 'DisplayName', '理论轨迹');
legend('接收信号点', '理论轨迹');
hold off;


% --- 图 3: 功率谱密度 (PSD) ---
figure('Name', 'AM 功率谱密度');
% 使用 pwelch 估计PSD
pwelch(y, [], [], [], fs_rf, 'centered', 'psd');
title('功率谱密度 (Power Spectral Density)');
% pwelch会自动处理坐标轴，但横轴是以Hz为单位的

% 手动标注关键频率点，以增强可读性
hold on;
ylim_vals = ylim;
% 标注载波
line([fc, fc], ylim_vals, 'Color', 'red', 'LineStyle', '--');
text(fc, ylim_vals(2)*0.9, sprintf('  载波\n  %.1f kHz', fc/1000), 'Color', 'red');
% 标注上边带
line([fc+fm1, fc+fm1], ylim_vals, 'Color', 'green', 'LineStyle', ':');
line([fc+fm2, fc+fm2], ylim_vals, 'Color', 'green', 'LineStyle', ':');
text(fc+fm2, ylim_vals(2)*0.6, '上边带 (USB)', 'Color', 'green');
% 标注下边带
line([fc-fm1, fc-fm1], ylim_vals, 'Color', 'magenta', 'LineStyle', ':');
line([fc-fm2, fc-fm2], ylim_vals, 'Color', 'magenta', 'LineStyle', ':');
text(fc-fm2-3000, ylim_vals(2)*0.6, '下边带 (LSB)', 'Color', 'magenta');
hold off;