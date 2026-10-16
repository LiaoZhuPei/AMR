% =========================================================================
% 主脚本：生成、对比并绘制QAM和APSK信号 (独立窗口版本)
% =========================================================================
clc, clear, close all

% --- 生成QAM信号 ---
[y_qam, L] = generateQAMSignal();

% --- 生成APSK信号 ---
[y_apsk, ~] = generateAPSKSignal(); % L应该相同

% --- 提取I路和Q路分量 ---
I_qam = real(y_qam);
Q_qam = imag(y_qam);

I_apsk = real(y_apsk);
Q_apsk = imag(y_apsk);

% --- 归一化信号到 [-1, 1] ---
I_qam_norm = normalize_signal(I_qam);
Q_qam_norm = normalize_signal(Q_qam);

I_apsk_norm = normalize_signal(I_apsk);
Q_apsk_norm = normalize_signal(Q_apsk);

% --- 绘图 (每个图在独立窗口) ---
time_axis = (0:L-1); % 创建时间轴

% 图1: I路信号原始对比 (独立窗口)
figure('Name', '1. I-Path Signal Comparison (Original)', 'NumberTitle', 'off');
plot(time_axis, I_qam, 'b-', 'DisplayName', 'QAM I-Path');
hold on;
plot(time_axis, I_apsk, 'r-', 'DisplayName', 'APSK I-Path');
hold off;
grid on;
title('1. I-Path Signal Comparison (Original)');
xlabel('Sample Index');
ylabel('Amplitude');
legend;

% 图2: Q路信号原始对比 (独立窗口)
figure('Name', '2. Q-Path Signal Comparison (Original)', 'NumberTitle', 'off');
plot(time_axis, Q_qam, 'b-', 'DisplayName', 'QAM Q-Path');
hold on;
plot(time_axis, Q_apsk, 'r-', 'DisplayName', 'APSK Q-Path');
hold off;
grid on;
title('2. Q-Path Signal Comparison (Original)');
xlabel('Sample Index');
ylabel('Amplitude');
legend;

% 图3: I路信号归一化对比 (独立窗口)
figure('Name', '3. I-Path Signal Comparison (Normalized)', 'NumberTitle', 'off');
plot(time_axis, I_qam_norm, 'b-', 'DisplayName', 'QAM I-Path (Normalized)');
hold on;
plot(time_axis, I_apsk_norm, 'r-', 'DisplayName', 'APSK I-Path (Normalized)');
hold off;
grid on;
title('3. I-Path Signal Comparison (Normalized to [-1, 1])');
xlabel('Sample Index');
ylabel('Normalized Amplitude');
legend;

% 图4: Q路信号归一化对比 (独立窗口)
figure('Name', '4. Q-Path Signal Comparison (Normalized)', 'NumberTitle', 'off');
plot(time_axis, Q_qam_norm, 'b-', 'DisplayName', 'QAM Q-Path (Normalized)');
hold on;
plot(time_axis, Q_apsk_norm, 'r-', 'DisplayName', 'APSK Q-Path (Normalized)');
hold off;
grid on;
title('4. Q-Path Signal Comparison (Normalized to [-1, 1])');
xlabel('Sample Index');
ylabel('Normalized Amplitude');
legend;

% =========================================================================
% 辅助函数区 (这部分代码保持不变)
% =========================================================================

function [y, L] = generateQAMSignal()
    % --- QAM 信号生成函数 ---
    RndPhase = true;

    ModType = 'qam';
    M_i = 32;
    sps_i = 4;
    phaserot = 0;
    ini_phase_i = 0;
    dfcTs_i = 0;
    rolloff = 0.35;
    symbol_order = 'gray';

    % 假设 Initialize_Modulation_NN 和 qammod_NN 存在
    [Mod_QAM] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

    if RndPhase
        Mod_QAM.ini_phase = 2*pi*rand(1,1);
    end

    L = 1024;
    SNR = 24;

    dts = 0:(sps_i-1);
    dt = dts(randi([1 length(dts)],[1 1]));

    delay = 0;
    Nsyms = ceil((delay+L+dt+1)/sps_i);
    data = randi([0 1],[1 Nsyms*log2(Mod_QAM.M)]);
    x = qammod_NN(Mod_QAM, data);
    y = awgn(x, SNR, 'measured');
    y = y(delay+dt+(1:L));
end

function [y, L] = generateAPSKSignal()
    % --- APSK 信号生成函数 ---
    RndPhase = true;

    ModType = 'apsk';
    M_i = 32;
    sps_i = 4;
    phaserot = 0;
    ini_phase_i = 0;
    dfcTs_i = 0;
    rolloff = 0.35;
    symbol_order = 'bin';

    % 假设 Initialize_Modulation_NN 和 apskmod_NN 存在
    [Mod_APSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
    
    if RndPhase
        Mod_APSK.ini_phase = 2*pi*rand(1,1);
    end

    L = 1024;
    SNR = 36;
    
    dts = 0:(sps_i-1);
    dt = dts(randi([1 length(dts)],[1 1]));

    delay = 0;
    Nsyms = ceil((delay+L+dt+1)/sps_i);
    data = randi([0 1],[1 Nsyms*log2(Mod_APSK.M)]);
    x = apskmod_NN(Mod_APSK, data);
    y = awgn(x, SNR, 'measured');
    y = y(delay+dt+(1:L));
end

function normalized_y = normalize_signal(y)
    % --- Min-Max 归一化函数，将信号缩放到 [-1, 1] ---
    min_val = min(y);
    max_val = max(y);
    range_val = max_val - min_val;

    if range_val == 0
        % 如果信号是常数，则返回全零或常数值
        normalized_y = zeros(size(y));
    else
        % 归一化到 [0, 1]，然后映射到 [-1, 1]
        normalized_y = 2 * (y - min_val) / range_val - 1;
    end
end