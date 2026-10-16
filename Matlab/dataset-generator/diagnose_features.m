function diagnose_features(iq_real, iq_sim)
% DIAGNOSE_FEATURES 特征分布直方图对比
% 作用：精准定位是哪个统计特征导致了 t-SNE 分离

    % 1. 切片设置
    slice_len = 1024;
    num_samples = 2000; % 取多一点样本看分布
    
    % 2. 提取特征 (复用之前的逻辑)
    [feats_r, ~] = extract_feats(iq_real, slice_len, num_samples);
    [feats_s, ~] = extract_feats(iq_sim,  slice_len, num_samples);
    
    % 3. 绘图对比
    feat_names = {'PAPR (dB)', 'Kurtosis', 'Skewness', 'Amp Variance'};
    
    figure('Position', [100, 100, 1200, 300], 'Color', 'w');
    sgtitle('实测(红) vs 仿真(蓝) 特征分布诊断');
    
    for i = 1:4
        subplot(1, 4, i);
        
        % 统一 bin 范围以便对比
        data_all = [feats_r(:,i); feats_s(:,i)];
        bin_edges = linspace(min(data_all), max(data_all), 50);
        
        % 画直方图
        histogram(feats_r(:,i), bin_edges, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5); hold on;
        histogram(feats_s(:,i), bin_edges, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
        
        title(feat_names{i});
        legend('Real', 'Sim'); grid on;
        
        % 计算重叠度 (Wasserstein 距离的简化版：均值差)
        diff_mean = mean(feats_r(:,i)) - mean(feats_s(:,i));
        xlabel(sprintf('Diff Mean: %.3f', diff_mean));
    end
    
    fprintf('诊断完成。请观察图形：\n');
    fprintf('1. 哪个图红蓝完全没重叠？那就是导致 t-SNE 分离的元凶。\n');
end

function [feats, labels] = extract_feats(long_signal, L, N)
    feats = zeros(N, 4);
    total = length(long_signal);
    idx = randi([1, total-L], N, 1);
    for i=1:N
        sig = long_signal(idx(i):idx(i)+L-1);
        % 必须去直流 + 归一化
        sig = sig - mean(sig); 
        p = mean(abs(sig).^2);
        if p>0, sig = sig/sqrt(p); end
        
        amp = abs(sig);
        feats(i,1) = 10*log10(max(amp.^2)/mean(amp.^2)); % PAPR
        feats(i,2) = kurtosis(amp); % Kurtosis
        feats(i,3) = skewness(amp); % Skewness
        feats(i,4) = var(amp);      % AmpVar
    end
    labels = ones(N,1);
end