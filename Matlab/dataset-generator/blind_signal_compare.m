function blind_signal_compare(iq_real, iq_sim)
% 输入:
%   iq_real: 实测的复数 IQ 数据 (列向量)
%   iq_sim:  仿真的复数 IQ 数据 (列向量)

    %% 1. 数据预处理 (至关重要)
    % 截取相同长度以便对比 (取两者较短的长度)
    min_len = min(length(iq_real), length(iq_sim));
    min_len = min(min_len, 100000); % 限制最大点数，防止绘图卡顿
    
    y_real = iq_real(1:min_len);
    y_sim  = iq_sim(1:min_len);
    
    % 去直流
    y_real = y_real - mean(y_real);
    y_sim  = y_sim  - mean(y_sim);
    
    % **关键步骤：功率归一化**
    % 将两者能量都缩放到 1，否则幅度分布无法对比
    y_real = y_real / sqrt(mean(abs(y_real).^2));
    y_sim  = y_sim  / sqrt(mean(abs(y_sim).^2));

    %% 2. 特征计算
    
    % A. 幅度统计 (Amplitude Statistics)
    amp_real = abs(y_real);
    amp_sim  = abs(y_sim);
    
    % 峰均比 (PAPR)
    papr_real = 10*log10(max(amp_real.^2) / mean(amp_real.^2));
    papr_sim  = 10*log10(max(amp_sim.^2) / mean(amp_sim.^2));
    
    % 峭度 (Kurtosis) - 衡量分布的"陡峭"程度
    kurt_real = kurtosis(amp_real);
    kurt_sim  = kurtosis(amp_sim);

    fprintf('--- 统计特征对比 ---\n');
    fprintf('指标\t\t实测信号\t仿真信号\t差异\n');
    fprintf('PAPR (dB)\t%.2f\t\t%.2f\t\t%.2f\n', papr_real, papr_sim, abs(papr_real-papr_sim));
    fprintf('Kurtosis\t%.2f\t\t%.2f\t\t%.2f\n', kurt_real, kurt_sim, abs(kurt_real-kurt_sim));

    %% 3. 可视化对比
    
    figure('Position', [100, 100, 1200, 800], 'Color', 'w');
    
    % --- 子图 1: 幅度直方图 (Amplitude PDF) ---
    % 这是调整非线性和滤波器的核心依据
    subplot(2, 2, 1);
    histogram(amp_real, 100, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2, 'EdgeColor', 'r');
    hold on;
    histogram(amp_sim, 100, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2, 'EdgeColor', 'b');
    legend('实测信号', '仿真信号');
    title('1. 幅度概率分布 (Amplitude PDF)');
    xlabel('归一化幅度'); ylabel('概率密度');
    grid on;
    
    % --- 子图 2: 功率谱密度 (PSD) ---
    % 这是调整带宽和 SNR 的依据
    subplot(2, 2, 2);
    [pxx_real, f_real] = pwelch(y_real, kaiser(1024, 5), 512, 1024, 'centered');
    [pxx_sim, f_sim]   = pwelch(y_sim,  kaiser(1024, 5), 512, 1024, 'centered');
    
    % 归一化频率轴 (-0.5 ~ 0.5)
    f_real = f_real / (2*pi); 
    f_sim  = f_sim  / (2*pi);
    
    plot(f_real, 10*log10(pxx_real), 'r', 'LineWidth', 1.5); hold on;
    plot(f_sim,  10*log10(pxx_sim),  'b', 'LineWidth', 1.5);
    legend('实测信号', '仿真信号');
    title('2. 功率谱密度 (PSD)');
    xlabel('归一化频率 (\times f_s)'); ylabel('功率 (dB/Hz)');
    grid on; ylim([-80 10]);
    
    % --- 子图 3: 星座图概览 (Constellation) ---
    % 观察相位噪声和 IQ 不平衡
    subplot(2, 2, 3);
    plot(y_real(1:2000), '.r', 'MarkerSize', 4); hold on;
    title('实测信号星座 (前2000点)');
    axis square; xlim([-2 2]); ylim([-2 2]); grid on;
    
    subplot(2, 2, 4);
    plot(y_sim(1:2000), '.b', 'MarkerSize', 4);
    title('仿真信号星座 (前2000点)');
    axis square; xlim([-2 2]); ylim([-2 2]); grid on;
    
    sgtitle('实测 vs 仿真：盲特征对比');

end