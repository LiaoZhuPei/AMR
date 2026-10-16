clc;
clear;
close all;

% --- 全局参数设置 ---
RndPhase = true;      % 是否使用随机初始相位
L = 1024;             % 每个突发的长度（采样点数）
sps_i = 4;            % 每个符号的采样点数 (samples per symbol)
rolloff = 0.35;       % 滚降因子
symbol_order = 'gray';% 符号映射顺序 ('gray' 或 'bin')
ModType = 'qam';

% --- 主要循环控制参数 ---
snr_levels_dB = [6, 14, 22, 30]; % 1. 四个信噪比等级
images_per_snr = 25;             % 每个信噪比下保存25张图像

% 创建一个文件夹来保存图像
output_dir = 'QAM_Waveform_Images';
if ~exist(output_dir, 'dir')
   mkdir(output_dir);
end

fprintf('开始生成图像...\n');

% --- 外层循环：遍历所有信噪比 ---
for snr_val = snr_levels_dB
    
    fprintf('正在处理 SNR = %d dB...\n', snr_val);
    
    % --- 内层循环：为每个信噪比生成指定数量的图像 ---
    for img_idx = 1:images_per_snr
        
        % --- 为每次迭代生成随机参数 ---
        % 随机初始相位
        if RndPhase
            ini_phase_i = 2*pi*rand(1,1);
        else
            ini_phase_i = 0;
        end
        
        % 随机采样时间偏移
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        
        % 计算所需的符号数 (两种调制方式使用相同的符号数)
        Nsyms = ceil((delay + L + dt + 1) / sps_i);

        % =================================================================
        % 1. 生成 16QAM 信号
        % =================================================================
        M_16QAM = 16;
        [Mod_16QAM] = Initialize_Modulation_NN(ModType, 'M', M_16QAM, 'sps', sps_i, ...
            'phaserot', 0, 'ini_phase', ini_phase_i, 'dfcTs', 0, 'beta', rolloff, ...
            'symbol_order', symbol_order);
        
        % 生成16QAM的随机比特流
        data_16QAM = randi([0 1], [1, Nsyms * log2(Mod_16QAM.M)]);
        
        % 调制
        x_16QAM = qammod_NN(Mod_16QAM, data_16QAM);
        
        % 添加高斯白噪声 (不进行归一化，'measured'会测量x的功率)
        y_16QAM_noisy = awgn(x_16QAM, snr_val, 'measured');
        
        % 截取突发信号
        y_16QAM_burst = y_16QAM_noisy(delay + dt + (1:L));

        % =================================================================
        % 2. 生成 64QAM 信号
        % =================================================================
        M_64QAM = 64;
        [Mod_64QAM] = Initialize_Modulation_NN(ModType, 'M', M_64QAM, 'sps', sps_i, ...
            'phaserot', 0, 'ini_phase', ini_phase_i, 'dfcTs', 0, 'beta', rolloff, ...
            'symbol_order', symbol_order);

        % 生成64QAM的随机比特流
        data_64QAM = randi([0 1], [1, Nsyms * log2(Mod_64QAM.M)]);
        
        % 调制
        x_64QAM = qammod_NN(Mod_64QAM, data_64QAM);
        
        % 添加高斯白噪声 (不进行归一化，'measured'会测量x的功率)
        y_64QAM_noisy = awgn(x_64QAM, snr_val, 'measured');
        
        % 截取突发信号
        y_64QAM_burst = y_64QAM_noisy(delay + dt + (1:L));

        % =================================================================
        % 3. 绘图与保存
        % =================================================================
        figure('Visible', 'off'); % 创建一个不可见的figure，以加快处理速度
        
        % --- 绘制I路波形 ---
        plot(real(y_16QAM_burst), 'LineWidth', 1.0);
        hold on;
        plot(real(y_64QAM_burst), 'LineWidth', 1.0, 'LineStyle', '-');
        
        % --- (可选) 绘制Q路波形 ---
        % 如果需要绘制Q路，请注释掉上面的plot，并取消注释下面的plot
        % plot(imag(y_16QAM_burst), 'LineWidth', 1.0);
        % hold on;
        % plot(imag(y_64QAM_burst), 'LineWidth', 1.0, 'LineStyle', '--');
        
        hold off;
        grid on;
        
        % 设置标题和标签
        title_str = sprintf('16QAM vs 64QAM (I-Channel) at SNR = %d dB, Image %d', snr_val, img_idx);
        % 如果切换到Q路，请修改这里的标题
        % title_str = sprintf('16QAM vs 64QAM (Q-Channel) at SNR = %d dB, Image %d', snr_val, img_idx);
        title(title_str);
        xlabel('Sample Index');
        ylabel('Amplitude');
        legend('16QAM', '64QAM');
        
        % 保存图像
        filename = fullfile(output_dir, sprintf('Comparison_SNR_%ddB_Img_%02d.png', snr_val, img_idx));
        saveas(gcf, filename);
        
        % 关闭当前figure，释放内存
        close(gcf);
    end
    
    fprintf('SNR = %d dB 的 %d 张图像已保存。\n', snr_val, images_per_snr);
end

fprintf('所有任务完成！总共生成了 %d 张图像。\n', length(snr_levels_dB) * images_per_snr);