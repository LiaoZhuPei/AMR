clc, clear, close all

% --- 生成QAM信号 ---
[y_qam, L] = generateQAMSignal();

% --- 生成APSK信号 ---
[y_apsk, ~] = generateAPSKSignal(); % L应该相同

amp_qam = abs(y_qam);
amp_apsk = abs(y_apsk);

figure('Name', 'Amplitude Distribution Analysis');
subplot(1, 2, 1);
histogram(amp_qam, 100, 'Normalization', 'pdf');
title('QAM Amplitude PDF');
xlabel('Amplitude');
ylabel('Probability Density');
grid on;

subplot(1, 2, 2);
histogram(amp_apsk, 100, 'Normalization', 'pdf');
title('APSK Amplitude PDF');
xlabel('Amplitude');
ylabel('Probability Density');
grid on;

phase_qam = angle(y_qam);
phase_apsk = angle(y_apsk);

% 计算相位差分
phase_diff_qam = diff(unwrap(phase_qam));
phase_diff_apsk = diff(unwrap(phase_apsk));

figure('Name', 'Phase Distribution Analysis');
subplot(1, 2, 1);
histogram(phase_diff_qam, 100, 'Normalization', 'pdf');
title('QAM Phase Difference PDF');
xlabel('Phase Change (radians)');
ylabel('Probability Density');
grid on;

subplot(1, 2, 2);
histogram(phase_diff_apsk, 100, 'Normalization', 'pdf');
title('APSK Phase Difference PDF');
xlabel('Phase Change (radians)');
ylabel('Probability Density');
grid on

% 计算幅度峰度
kurt_amp_qam = kurtosis(abs(y_qam));
kurt_amp_apsk = kurtosis(abs(y_apsk));

fprintf('Amplitude Kurtosis -> QAM: %.4f, APSK: %.4f\n', kurt_amp_qam, kurt_amp_apsk);

% 计算I路分量的峰度
kurt_i_qam = kurtosis(real(y_qam));
kurt_i_apsk = kurtosis(real(y_apsk));

fprintf('I-Path Kurtosis -> QAM: %.4f, APSK: %.4f\n', kurt_i_qam, kurt_i_apsk)

[acf_qam, lags_qam] = xcorr(y_qam, 100, 'normalized'); % 计算前100个延迟
[acf_apsk, lags_apsk] = xcorr(y_apsk, 100, 'normalized');

figure('Name', 'Autocorrelation Analysis');
plot(lags_qam, abs(acf_qam), 'b-', 'DisplayName', 'QAM');
hold on;
plot(lags_apsk, abs(acf_apsk), 'r-', 'DisplayName', 'APSK');
hold off;
grid on;
title('Autocorrelation Comparison');
xlabel('Lag');
ylabel('Normalized Correlation');
legend;

function [y, L] = generateQAMSignal()
    % --- QAM 信号生成函数 ---
    RndPhase = true;

    ModType = 'qam';
    M_i = 16;
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
    SNR = 32;

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
    M_i = 16;
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
    SNR = 32;
    
    dts = 0:(sps_i-1);
    dt = dts(randi([1 length(dts)],[1 1]));

    delay = 0;
    Nsyms = ceil((delay+L+dt+1)/sps_i);
    data = randi([0 1],[1 Nsyms*log2(Mod_APSK.M)]);
    x = apskmod_NN(Mod_APSK, data);
    y = awgn(x, SNR, 'measured');
    y = y(delay+dt+(1:L));
end