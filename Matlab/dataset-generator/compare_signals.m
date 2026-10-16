function compare_signals(real_sig, sim_sig, fs)
% 对比实测和仿真信号
    
    % 截取相同长度
    L = min(length(real_sig), length(sim_sig));
    L = min(L, 100000); % 限制最大点数
    
    r = real_sig(1:L); r = r(:) - mean(r); r = r/sqrt(mean(abs(r).^2));
    s = sim_sig(1:L);  s = s(:) - mean(s); s = s/sqrt(mean(abs(s).^2));
    
    figure('Position', [50, 50, 1200, 800], 'Color', 'w');
    sgtitle('实测 vs 仿真深度对比');
    
    %% 1. 功率谱密度 (PSD) - 使用 Blackman-Harris 窗
    subplot(2,2,1);
    nfft = 1024;
    window = blackmanharris(nfft); % [重要] 极低旁瓣窗函数
    [pr, fr] = pwelch(r, window, nfft/2, nfft, fs, 'centered');
    [ps, fs_ax] = pwelch(s, window, nfft/2, nfft, fs, 'centered');
    
    plot(fr/1e6, 10*log10(pr), 'r', 'LineWidth', 1.5); hold on;
    plot(fs_ax/1e6, 10*log10(ps), 'b', 'LineWidth', 1.5);
    title('PSD (频谱锯齿与底噪)'); xlabel('MHz'); ylabel('dB/Hz');
    legend('实测', '仿真'); grid on; ylim([-120 0]);
    
    %% 2. 幅度概率分布 (PDF)
    subplot(2,2,2);
    histogram(abs(r), 100, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2, 'EdgeColor', 'r'); hold on;
    histogram(abs(s), 100, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'LineWidth', 2, 'EdgeColor', 'b');
    title('Amplitude PDF (非线性特征)'); grid on;
    
    %% 3. CCDF (PAPR)
    subplot(2,2,3);
    [yr, xr] = calc_ccdf(r);
    [ys, xs] = calc_ccdf(s);
    semilogy(xr, yr, 'r', 'LineWidth', 2); hold on;
    semilogy(xs, ys, 'b', 'LineWidth', 2);
    title('CCDF (峰值特征)'); xlabel('dB above Average'); grid on; xlim([0 10]);
    
    %% 4. 星座图
    subplot(2,2,4);
    plot(r(1:2000), '.r'); hold on;
    plot(s(1:2000), '.b');
    title('星座图 (相位噪声与失真)'); axis square; xlim([-2 2]); ylim([-2 2]); grid on;
end

function [y, x] = calc_ccdf(sig)
    p = abs(sig).^2; p_avg = mean(p);
    x_raw = 10*log10(p / p_avg);
    [x, ~] = sort(x_raw);
    y = 1 - (1:length(x))'/length(x);
end