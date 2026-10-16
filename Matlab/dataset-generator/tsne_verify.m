function tsne_verify(iq_real_data, iq_sim_data)
% 使用 t-SNE 验证实测数据与仿真数据的分布相似性
% 输入:
%   iq_real_data: 实测复数数据 (长向量)
%   iq_sim_data:  仿真复数数据 (长向量)

    %% 1. 数据切片 (Slicing)
    % 将长信号切成多个短样本，模拟神经网络的输入
    slice_len = 1024; % 每个样本长度
    num_samples = 500; % 提取样本数量 (越多越准，但慢)
    
    % 提取实测样本
    real_slices = extract_slices(iq_real_data, slice_len, num_samples);
    
    % 提取仿真样本
    sim_slices = extract_slices(iq_sim_data, slice_len, num_samples);
    
    %% 2. 特征提取 (Feature Extraction)
    % 提取高阶统计量特征，这些特征对调制识别最敏感
    fprintf('正在提取特征...\n');
    feat_real = get_features(real_slices);
    feat_sim  = get_features(sim_slices);
    
    %% 3. 数据合并与标准化
    % 合并特征矩阵: [N_real + N_sim, Num_Features]
    X = [feat_real; feat_sim];
    
    % 创建标签: 1=实测, 2=仿真
    labels = [ones(num_samples, 1); 2 * ones(num_samples, 1)];
    label_strs = {'Real', 'Sim'};
    
    % Z-Score 标准化 (非常重要，否则量纲不同会主导距离)
    X_norm = zscore(X);
    
    %% 4. 运行 t-SNE 降维
    fprintf('正在运行 t-SNE (可能需要几秒钟)...\n');
    % 使用 barneshut 算法加速，并将维度降至 2
    % Perplexity: 困惑度，通常 30-50
    Y = tsne(X_norm, 'Algorithm', 'barneshut', 'NumDimensions', 2, 'Perplexity', 30);
    
    %% 5. 可视化结果
    figure('Position', [100, 100, 1000, 500], 'Color', 'w');
    
    % 子图 1: t-SNE 散点图
    subplot(1, 2, 1);
    gscatter(Y(:,1), Y(:,2), labels, 'rb', '..', 15);
    title('t-SNE 特征分布可视化');
    xlabel('Dimension 1'); ylabel('Dimension 2');
    legend('实测数据 (Real)', '仿真数据 (Sim)', 'Location', 'best');
    grid on; axis square;
    
    % 子图 2: 特征箱线图对比 (验证具体特征差异)
    subplot(1, 2, 2);
    % 挑选几个关键特征画箱线图
    % 特征索引: 1=PAPR, 2=Kurtosis, 3=Skewness, 4=AmpVar, 5=PhaseVar
    feature_names = {'PAPR', 'Kurtosis', 'Skewness', 'Amp Var'};
    selected_feats = [1, 2, 3, 4];
    
    % 为了画图方便，重组数据
    group_labels = [];
    data_vector = [];
    x_labels = {};
    
    for i = 1:length(selected_feats)
        idx = selected_feats(i);
        % 实测数据
        data_vector = [data_vector; X_norm(1:num_samples, idx)];
        group_labels = [group_labels; repmat({['Real-' feature_names{i}]}, num_samples, 1)];
        % 仿真数据
        data_vector = [data_vector; X_norm(num_samples+1:end, idx)];
        group_labels = [group_labels; repmat({['Sim-' feature_names{i}]}, num_samples, 1)];
    end
    
    boxplot(data_vector, group_labels, 'Symbol', ''); % 不画离群点
    title('关键特征分布对比 (标准化后)');
    xtickangle(45); grid on;
    
    fprintf('完成！请观察左图：红蓝点是否混合？\n');
    fprintf('如果红蓝点分得太开，说明特征分布差异大。\n');
end

%% 辅助函数：切片
function slices = extract_slices(long_signal, L, N)
    slices = zeros(N, L);
    total_len = length(long_signal);
    
    % 随机切片
    start_indices = randi([1, total_len - L], N, 1);
    
    for i = 1:N
        segment = long_signal(start_indices(i) : start_indices(i) + L - 1);
        % 归一化每个切片
        segment = segment - mean(segment);
        segment = segment / sqrt(mean(abs(segment).^2));
        slices(i, :) = segment.'; % 转置为行向量
    end
end

%% 辅助函数：特征提取
function feats = get_features(slices)
    [N, ~] = size(slices);
    % 特征列表:
    % 1. 峰均比 (PAPR)
    % 2. 幅度峭度 (Kurtosis) - 衡量非高斯性
    % 3. 幅度偏度 (Skewness) - 衡量分布不对称性
    % 4. 幅度方差 (Variance of Amplitude)
    % 5. 相位方差 (Variance of Phase - 差分后)
    % 6. 零中心归一化瞬时幅度谱密度的最大值 (gamma_max)
    
    feats = zeros(N, 6);
    
    for i = 1:N
        sig = slices(i, :);
        amp = abs(sig);
        phi = angle(sig);
        
        % 1. PAPR (dB)
        p_peak = max(amp.^2);
        p_avg = mean(amp.^2);
        feats(i, 1) = 10*log10(p_peak / p_avg);
        
        % 2. Kurtosis (幅度)
        feats(i, 2) = kurtosis(amp);
        
        % 3. Skewness (幅度)
        feats(i, 3) = skewness(amp);
        
        % 4. Amplitude Variance (归一化)
        feats(i, 4) = var(amp); % 因为功率已归一化，这就代表幅度波动
        
        % 5. Phase Variance (差分相位，去除频偏影响)
        d_phi = diff(unwrap(phi));
        feats(i, 5) = var(d_phi);
        
        % 6. 简单的频域特征: 能量集中度
        f_spec = abs(fft(sig));
        feats(i, 6) = max(f_spec) / sum(f_spec);
    end
end