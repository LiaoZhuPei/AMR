clc;
clear;
close all;

% --- 全局参数设置 ---
RndPhase = true;      % 是否使用随机初始相位
L = 1024;             % 每个突发的总长度（采样点数）
sps_i = 4;            % 每个符号的采样点数 (samples per symbol)
rolloff = 0.35;       % 滚降因子
symbol_order = 'gray';% 符号映射顺序 ('gray' 或 'bin')
ModType = 'qam';

% --- 主要循环控制参数 ---
snr_levels_dB = [6, 14, 22, 30]; % 1. 四个信噪比等级
images_per_snr = 25;             % 每个信噪比下保存25张图像

% --- 绘图参数 ---
% 新增：设置要在图表中显示的采样点数量，以“放大”X轴
samples_to_plot = 200; 
if samples_to_plot > L
    error('samples_to_plot 不能大于总长度 L');
end


% 创建一个文件夹来保存图像
output_dir = 'QAM_Waveform_Images_Zoomed';
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
        if RndPhase
            ini_phase_i = 2*pi*rand(1,1);
        else
            ini_phase_i = 0;
        end
        
        dts = 0:(sps_i-1);
        dt = dts(randi([1 length(dts)],[1 1]));
        delay = 0;
        
        Nsyms = ceil((delay + L + dt + 1) / sps_i);

        % =================================================================
        % 1. 生成 16QAM 信号
        % =================================================================
        M_16QAM = 16;
        [Mod_16QAM] = Initialize_Modulation_NN(ModType, 'M', M_16QAM, 'sps', sps_i, ...
            'phaserot', 0, 'ini_phase', ini_phase_i, 'dfcTs', 0, 'beta', rolloff, ...
            'symbol_order', symbol_order);
        
        data_16QAM = randi([0 1], [1, Nsyms * log2(Mod_16QAM.M)]);
        x_16QAM = qammod_NN(Mod_16QAM, data_16QAM);
        y_16QAM_noisy = awgn(x_16QAM, snr_val, 'measured');
        y_16QAM_burst = y_16QAM_noisy(delay + dt + (1:L));

        % =================================================================
        % 2. 生成 64QAM 信号
        % =================================================================
        M_64QAM = 64;
        [Mod_64QAM] = Initialize_Modulation_NN(ModType, 'M', M_64QAM, 'sps', sps_i, ...
            'phaserot', 0, 'ini_phase', ini_phase_i, 'dfcTs', 0, 'beta', rolloff, ...
            'symbol_order', symbol_order);

        data_64QAM = randi([0 1], [1, Nsyms * log2(Mod_64QAM.M)]);
        x_64QAM = qammod_NN(Mod_64QAM, data_64QAM);
        y_64QAM_noisy = awgn(x_64QAM, snr_val, 'measured');
        y_64QAM_burst = y_64QAM_noisy(delay + dt + (1:L));

        % =================================================================
        % 3. 绘图与保存
        % =================================================================
        figure('Visible', 'off'); 
        
        % --- 绘制I路波形 (只绘制前 'samples_to_plot' 个点) ---
        plot(real(y_16QAM_burst(1:samples_to_plot)), 'LineWidth', 1.2);
        hold on;
        plot(real(y_64QAM_burst(1:samples_to_plot)), 'LineWidth', 1.2, 'LineStyle', '-');
        
        % --- (可选) 绘制Q路波形 (只绘制前 'samples_to_plot' 个点) ---
        % plot(imag(y_16QAM_burst(1:samples_to_plot)), 'LineWidth', 1.2);
        % hold on;
        % plot(imag(y_64QAM_burst(1:samples_to_plot)), 'LineWidth', 1.2, 'LineStyle', '--');
        
        hold off;
        grid on;
        
        title_str = sprintf('16QAM vs 64QAM (I-Channel) at SNR = %d dB, Image %d', snr_val, img_idx);
        title(title_str);
        xlabel(sprintf('Sample Index (First %d Samples)', samples_to_plot));
        ylabel('Amplitude');
        legend('16QAM', '64QAM', 'Location', 'northeast'); % 将图例放在右上角
        
        % 另一种放大X轴的方法是使用 xlim，可以取消下面的注释来试用
        % xlim([1, samples_to_plot]); 
        
        % 保存图像
        filename = fullfile(output_dir, sprintf('Comparison_SNR_%ddB_Img_%02d.png', snr_val, img_idx));
        saveas(gcf, filename);
        
        close(gcf);
    end
    
    fprintf('SNR = %d dB 的 %d 张图像已保存。\n', snr_val, images_per_snr);
end

fprintf('所有任务完成！总共生成了 %d 张图像。\n', length(snr_levels_dB) * images_per_snr);