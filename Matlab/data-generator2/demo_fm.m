% 主 FM 仿真脚本

clc, clear, close all

%% 1. 仿真参数设置
SNR = 25;               % 信噪比 (dB)
RndPhase = true;        % 是否使用随机初始相位

% --- FM 调制器参数 ---
ModType = 'fm';
h_i = 0.2;                % 调制指数 (β or h)。h>1 为宽带FM (WBFM)
sps_i = 4;             % 上采样因子。FM需要更高的sps以避免混叠
ini_phase_i = 0;        % 默认初始相位

% --- 基带和载波频率参数 ---
fs_baseband = 6e3;      % 基带信号的采样率 (Hz), 例如 6 kHz
fm_max = fs_baseband / 2; % 基带信号最大频率 (3 kHz)

% 根据卡森法则估算FM带宽并设置合理的载波和输出采样率
delta_f = h_i * fm_max; % 最大频偏 = 5 * 3kHz = 15 kHz
carson_bw = 2 * (delta_f + fm_max); % 卡森带宽 = 2 * (15+3) = 36 kHz

fc = 5e3;              % 载波频率 (Hz), 例如 48 kHz
fs_rf = fs_baseband * sps_i; % 输出采样率 = 6k * 16 = 96 kHz

% 确保输出采样率满足奈奎斯特准则
if fs_rf < (fc + carson_bw/2) * 2
    warning('输出采样率可能不足以避免混叠！');
end
fprintf('调制指数 h = %d\n', h_i);
fprintf('最大频偏 Δf = %.1f kHz\n', delta_f/1000);
fprintf('卡森法则估算带宽 ≈ %.1f kHz\n', carson_bw/1000);

% 计算归一化的载波频率
dfc_div_fs_i = fc / fs_rf; 

% 初始化FM调制器结构体
[Mod_FM] = Initialize_Modulation_NN(ModType, 'h', h_i, 'sps', sps_i, ...
    'ini_phase', ini_phase_i, 'dfc_div_fs', dfc_div_fs_i);

% 如果设置了随机相位，则覆盖默认值
if RndPhase
    Mod_FM.ini_phase = 2 * pi * rand(1, 1);
end

%% 2. 生成基带消息信号
% 使用单一正弦波，以便清晰地观察时域频率变化和频域边带
L_baseband = 1024;
T_baseband = 1 / fs_baseband;
t_baseband = (0:L_baseband-1) * T_baseband;

fm_tone = 1e3; % 基带正弦波频率 (1 kHz)
m_t = sin(2*pi*fm_tone*t_baseband);

%% 3. 调用FM调制器并模拟信道
% 调用调制函数生成理想的FM信号
x = fmmod_NN(Mod_FM, m_t);

% 通过AWGN信道添加噪声
y = awgn(x, SNR, 'measured');

%% 4. 可视化部分

L_rf = length(y);
t_rf = (0:L_rf-1) / fs_rf;

% --- 图 1: 完整的 FM IQ 时域波形 ---
figure('Name', 'FM IQ 时域波形');

% 绘制 I 分量 (实部)
plot(t_rf, real(y), 'b-', 'DisplayName', 'I 分量 (实部)'); 
hold on;

% 绘制 Q 分量 (虚部)
plot(t_rf, imag(y), 'g-', 'DisplayName', 'Q 分量 (虚部)'); % 使用绿色 'g-'

% 叠加归一化的基带信号，以对比频率变化
plot(t_rf, resample(m_t, sps_i, 1), 'r--', 'LineWidth', 1.5, 'DisplayName', '基带信号');

grid on;
title(sprintf('FM IQ 时域波形 (h = %d, SNR = %d dB)', h_i, SNR));
xlabel('时间 (s)');
ylabel('幅度');
legend('show'); % 'DisplayName' 属性会自动创建图例
xlim([0, 3/fm_tone]); % 放大显示几个周期的波形
hold off;

% --- 图 2: 星座图轨迹 ---
figure('Name', 'FM 星座图');
% 对于FM，信号点应在一个圆周上移动
scatter(real(y), imag(y), 10, 'filled', 'MarkerFaceAlpha', 0.1);
grid on;
axis square;
title('FM 星座图轨迹 (恒包络)');
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
lim_max = max(abs([xlim, ylim])) * 1.1;
axis([-lim_max, lim_max, -lim_max, lim_max]);
hold on;
% 绘制理论上的圆形轨迹
th = 0:pi/50:2*pi;
plot(cos(th), sin(th), 'r-', 'LineWidth', 2, 'DisplayName', '理论轨迹');
legend('接收信号点', '理论轨迹');
hold off;

% --- 图 3: 功率谱密度 (PSD) ---
figure('Name', 'FM 功率谱密度');
pwelch(y, [], [], [], fs_rf, 'centered', 'psd');
title(sprintf('宽带FM功率谱密度 (h = %d)', h_i));
hold on;
ylim_vals = ylim;
% 标注载波
line([fc, fc], ylim_vals, 'Color', 'red', 'LineStyle', '--');
text(fc, ylim_vals(2)*0.9, sprintf('  载波\n  %.1f kHz', fc/1000), 'Color', 'red');
% 标注卡森法则带宽
carson_half_bw = carson_bw / 2;
line([fc-carson_half_bw, fc-carson_half_bw], ylim_vals, 'Color', 'magenta', 'LineStyle', '--');
line([fc+carson_half_bw, fc+carson_half_bw], ylim_vals, 'Color', 'magenta', 'LineStyle', '--');
text(fc+carson_half_bw, ylim_vals(2)*0.7, sprintf('卡森带宽\n(%.1f kHz)', carson_bw/1000), 'Color', 'magenta');
hold off;