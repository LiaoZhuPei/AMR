clc, clear, close all

%% --- 1. 设置仿真和绘图参数 ---
% 这里的参数保持不变，用于生成信号
num_samples_per_snr = 20; 
snr_range_db = 12:2:36;

% --- 新增参数：用于方法B的信号尺寸 ---
MAX_LEN = 500;            % 最终网络输入的固定长度 (画布大小)
LONG_SIG_LEN = 2048;      % 用于截取的长信号流的长度
MIN_SIGNAL_LEN = 500;     % 截取信号的最小长度
MAX_SIGNAL_LEN = 500;     % 截取信号的最大长度

% --- 绘图保存设置 ---
output_plot_folder = 'OQPSK_Plots'; % 图片保存文件夹
if ~exist(output_plot_folder, 'dir')
   mkdir(output_plot_folder);
end

max_plots = 100;    % 限制只保存前100张图片
plot_counter = 0;   % 当前绘图计数器

%% --- 2. 主循环，生成和绘制数据 ---
fprintf('开始生成OQPSK信号并绘图 (限制前 %d 张)...\n', max_plots);

stop_flag = false; % 用于跳出双重循环的标志

for snr_db = snr_range_db
    if stop_flag, break; end % 如果达到数量限制，跳出外层循环
    
    fprintf('正在处理 SNR = %d dB...\n', snr_db);
    
    for sample_idx = 1:num_samples_per_snr
        % =================================================================
        % --- 核心生成逻辑 (保持不变) ---
        % =================================================================
        
        % --- a. 生成一个长信号流 ---
        RndPhase = true;
        ModType = 'oqpsk';
        M_i = 4;
        sps_i = 4;
        phaserot = 0;
        ini_phase_i = 0;
        dfcTs_i = 0;
        rolloff = 0.45;
        symbol_order = 'gray';
        
        % 注意：此处假设你当前目录下有 Initialize_Modulation_NN 和 oqpskmod_NN 函数
        [Mod_OQPSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
        if RndPhase
            Mod_OQPSK.ini_phase = 2*pi*rand(1,1);
        end
        
        SNR = snr_db;
        
        Nsyms_long = ceil(LONG_SIG_LEN / sps_i);
        data_long = randi([0 1],[1 Nsyms_long*log2(Mod_OQPSK.M)]);
        x_long = oqpskmod_NN(Mod_OQPSK, data_long); 
        
        % --- b. 随机截取 ---
        signal_len = randi([MIN_SIGNAL_LEN, MAX_SIGNAL_LEN]);
        
        if length(x_long) > signal_len
            crop_start = randi([1, length(x_long) - signal_len + 1]);
            x_cropped = x_long(crop_start : crop_start + signal_len - 1);
        else
            x_cropped = x_long;
            signal_len = length(x_long);
        end

        % --- c. 放入画布 ---
        canvas = zeros(1, MAX_LEN, 'like', 1i); 
        start_pos = randi([1, MAX_LEN - signal_len + 1]);
        canvas(start_pos : start_pos + signal_len - 1) = x_cropped;

        % --- d. 加噪 ---
        y_final = awgn(canvas, SNR, 'measured');
        
        % =================================================================
        % --- 绘图与保存逻辑 (修改部分) ---
        % =================================================================
        
        plot_counter = plot_counter + 1;
        
        % 创建不可见的 figure 以加快处理速度
        h_fig = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
        
        % 1. 绘制星座图
        subplot(2, 1, 1);
        plot(real(y_final), imag(y_final), '.', 'MarkerSize', 8);
        grid on; axis square;
        title(sprintf('Constellation Diagram (SNR=%d dB)', snr_db));
        xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
        xlim([-2 2]); ylim([-2 2]); % 固定坐标轴范围以便观察
        
        % 2. 绘制 IQ 时域波形
        subplot(2, 1, 2);
        time_axis = 1:MAX_LEN;
        plot(time_axis, real(y_final), 'b', 'DisplayName', 'I (Real)');
        hold on;
        plot(time_axis, imag(y_final), 'r--', 'DisplayName', 'Q (Imag)');
        hold off;
        grid on;
        legend('show');
        title(sprintf('IQ Waveform (Length=%d)', MAX_LEN));
        xlabel('Sample Index'); ylabel('Amplitude');
        ylim([-2.5 2.5]); % 固定坐标轴范围
        
        % --- 保存图片 ---
        % 文件名格式: OQPSK_序号_SNR.png
        snr_str = sprintf('%ddB', snr_db);
        if snr_db < 0, snr_str = sprintf('neg%ddB', abs(snr_db)); end
        
        filename_img = fullfile(output_plot_folder, ...
            sprintf('OQPSK_%03d_%s.png', plot_counter, snr_str));
        
        saveas(h_fig, filename_img);
        close(h_fig); % 关闭图形句柄释放内存
        
        fprintf('已保存图片: %s (%d/%d)\n', filename_img, plot_counter, max_plots);
        
        % --- 检查是否达到数量限制 ---
        if plot_counter >= max_plots
            stop_flag = true;
            break; % 跳出内层循环
        end
    end
end

fprintf('处理完成！共生成 %d 张图片，保存在 "%s" 文件夹下。\n', plot_counter, output_plot_folder);