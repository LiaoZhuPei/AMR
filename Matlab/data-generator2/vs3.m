clc, clear, close all

% --- 生成QAM信号 ---
[y_qam16, L] = generateQAMSignal16();

% --- 生成APSK信号 ---
[y_qam64, ~] = generateQAMSignal64(); % L应该相同

amp_qam16 = abs(y_qam16);
amp_qam64 = abs(y_qam64);

figure('Name', 'Amplitude Distribution Analysis');
subplot(1, 2, 1);
histogram(amp_qam16, 100, 'Normalization', 'pdf');
title('QAM16 Amplitude PDF');
xlabel('Amplitude');
ylabel('Probability Density');
grid on;

subplot(1, 2, 2);
histogram(amp_qam64, 100, 'Normalization', 'pdf');
title('QAM64 Amplitude PDF');
xlabel('Amplitude');
ylabel('Probability Density');
grid on;

phase_qam16 = angle(y_qam16);
phase_qam64 = angle(y_qam64);

% 计算相位差分
phase_diff_qam16 = diff(unwrap(phase_qam16));
phase_diff_qam64 = diff(unwrap(phase_qam64));

figure('Name', 'Phase Distribution Analysis');
subplot(1, 2, 1);
histogram(phase_diff_qam16, 100, 'Normalization', 'pdf');
title('QAM16 Phase Difference PDF');
xlabel('Phase Change (radians)');
ylabel('Probability Density');
grid on;

subplot(1, 2, 2);
histogram(phase_diff_qam64, 100, 'Normalization', 'pdf');
title('QAM64 Phase Difference PDF');
xlabel('Phase Change (radians)');
ylabel('Probability Density');
grid on

% 计算幅度峰度
kurt_amp_qam16 = kurtosis(abs(y_qam16));
kurt_amp_qam64 = kurtosis(abs(y_qam64));

fprintf('Amplitude Kurtosis -> QAM16: %.4f, QAM64: %.4f\n', kurt_amp_qam16, kurt_amp_qam64);

% 计算I路分量的峰度
kurt_i_qam16 = kurtosis(real(y_qam16));
kurt_i_qam64 = kurtosis(real(y_qam64));

fprintf('I-Path Kurtosis -> QAM16: %.4f, QAM64: %.4f\n', kurt_i_qam16, kurt_i_qam64)

[acf_qam16, lags_qam16] = xcorr(y_qam16, 100, 'normalized'); % 计算前100个延迟
[acf_qam64, lags_qam64] = xcorr(y_qam64, 100, 'normalized');

figure('Name', 'Autocorrelation Analysis');
plot(lags_qam16, abs(acf_qam16), 'b-', 'DisplayName', 'QAM16');
hold on;
plot(lags_qam64, abs(acf_qam64), 'r-', 'DisplayName', 'QAM64');
hold off;
grid on;
title('Autocorrelation Comparison');
xlabel('Lag');
ylabel('Normalized Correlation');
legend;

function [y, L] = generateQAMSignal16()
    % --- QAM 信号生成函数 ---
    RndPhase = true;

    ModType = 'qam';
    M_i = 16;
    sps_i = 2;
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

function [y, L] = generateQAMSignal64()
    % --- QAM 信号生成函数 ---
    RndPhase = true;

    ModType = 'qam';
    M_i = 64;
    sps_i = 2;
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
