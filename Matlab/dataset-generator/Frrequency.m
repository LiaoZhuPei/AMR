% MATLAB 仿真 AM 频谱分析

clear;
close all;
clc;

%% 1. 定义仿真参数
fs = 100e3;         % 采样频率 (Hz) - 100 kHz
T = 1/fs;           % 采样周期
L = 2^15;           % 信号长度，取2的幂次方以优化FFT计算
t = (0:L-1)*T;      % 时间向量

fc = 10e3;          % 载波频率 (Hz) - 10 kHz
Ac = 1;             % 载波幅度

% 基带调制信号 (两个正弦波的叠加，模拟更复杂的信号)
fm1 = 1e3;          % 基带信号频率 1 (Hz) - 1 kHz
fm2 = 2.5e3;        % 基带信号频率 2 (Hz) - 2.5 kHz
Am_max = 0.8;       % 基带信号最大幅度，保证调制指数小于1 (Am_max < Ac)

m_t = Am_max * (0.6 * sin(2*pi*fm1*t) + 0.4 * cos(2*pi*fm2*t)); % 基带信号

%% 2. 生成信号
% 载波信号
c_t = Ac * cos(2*pi*fc*t);

% AM 调制信号 (标准AM，也叫DSB-LC，带载波的双边带)
% s_AM(t) = [Ac + m(t)] * cos(2*pi*fc*t)
s_AM_t = (Ac + m_t) .* c_t;

%% 3. 计算并绘制频谱

% 定义一个匿名函数句柄，方便后续调用
plot_spectrum = @(signal_t, fs, L, title_str, x_lims, y_lims) ...
    figure_plot_spectrum(signal_t, fs, L, title_str, x_lims, y_lims);

% 绘制基带信号频谱
plot_spectrum(m_t, fs, L, '1. Baseband Signal Spectrum (m(t))', ...
              [0, max(fm1, fm2)*2/1000 + 1], [0, Am_max*0.7]);
          
% 绘制载波信号频谱
plot_spectrum(c_t, fs, L, '2. Carrier Signal Spectrum (c(t))', ...
              [fc/1000 - 1, fc/1000 + 1], [0, Ac*1.1]);
          
% 绘制AM调制信号频谱
plot_spectrum(s_AM_t, fs, L, '3. AM Modulated Signal Spectrum (s_AM(t))', ...
              [0, fc/1000 + max(fm1,fm2)/1000 + 2], [0, Ac*1.1]);
          
% 调整第三个图的X轴范围以更好地观察边带
figure(3);
xlim([fc/1000 - max(fm1,fm2)/1000 - 1, fc/1000 + max(fm1,fm2)/1000 + 1]);
title('3. AM Modulated Signal Spectrum (s_AM(t)) - Zoomed In');

%% 4. 时域波形可视化（可选）
figure;
subplot(3,1,1);
plot(t, m_t);
title('Baseband Signal m(t)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([0, 0.005]); % 仅显示一小段，以便看清波形

subplot(3,1,2);
plot(t, c_t);
title('Carrier Signal c(t)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([0, 0.005]);

subplot(3,1,3);
plot(t, s_AM_t);
title('AM Modulated Signal s_AM(t)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([0, 0.005]);

% 绘制包络线（用于验证AM调制效果）
hold on;
plot(t, Ac + m_t, 'r--', 'LineWidth', 1.5); % 上包络
plot(t, -(Ac + m_t), 'r--', 'LineWidth', 1.5); % 下包络 (对称)
legend('AM Signal', 'Envelope');
hold off;


% ===== 主脚本逻辑结束，以下为局部函数定义 =====
% ===== All executable code must be ABOVE this line =====

function figure_plot_spectrum(signal_t, fs, L, title_str, x_lims, y_lims)
    Y = fft(signal_t);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1); % 转换为单边谱，非DC和非Nyquist频率乘以2
    f = fs*(0:(L/2))/L;

    figure;
    plot(f/1000, P1); % 频率单位转换为 kHz
    title(title_str);
    xlabel('Frequency (kHz)');
    ylabel('Amplitude');
    grid on;
    if ~isempty(x_lims)
        xlim(x_lims);
    end
    if ~isempty(y_lims)
        ylim(y_lims);
    end
end