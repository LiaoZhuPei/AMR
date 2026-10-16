% =========================================================================
%                generate_data_from_iq.m
%       (一个完全用MATLAB实现的、用于处理.iq文件的数据生成脚本)
% =========================================================================
clc, clear, close all;

%% --- 1. 配置区 (与您的Python脚本完全对应) ---

% --- 输入配置 ---
IQ_FILE_PATH = 'E:/Project/AMR-Benchmark/ASK2-03-24-25-10h56m09s501.iq';
% !! 关键 !!: 指定.iq文件的数据类型。
% MATLAB中的名称: 'int16', 'single' (对应np.float32), 'int8'
DATA_TYPE = 'int16'; 

% --- 数据处理参数 ---
SAMPLES_TO_DISCARD = 10000;  % 丢弃的起始样本数
SEGMENT_LENGTH = 1024;       % 每个 .mat 文件的长度
STRIDE = SEGMENT_LENGTH;     % 步长 (无重叠)
PERFORM_NORMALIZATION = false; % !! 新增开关 !!: 是否执行归一化
                               % 设置为false以保留原始整数幅度

% --- 输出配置 ---
MAT_OUTPUT_DIR = 'output_mat_files_ASK2_matlab';
PLOT_OUTPUT_DIR = 'output_plot_reports_ASK2_matlab';
NUM_PLOTS_TO_SAVE = 100;

% --- .mat 文件中数据集的名称 ---
MAT_DATASET_NAME = 'data_save'; % 在.mat文件中保存数据的变量名

%% --- 2. 准备工作 ---

% 创建输出文件夹
if ~exist(MAT_OUTPUT_DIR, 'dir'), mkdir(MAT_OUTPUT_DIR); end
if ~exist(PLOT_OUTPUT_DIR, 'dir'), mkdir(PLOT_OUTPUT_DIR); end

%% --- 3. 主执行程序 ---
try
    % --- 步骤 A: 加载和清洗数据 ---
    fprintf('步骤 1: 正在从 "%s" 加载和清洗数据...\n', IQ_FILE_PATH);
    
    % 打开并读取二进制 .iq 文件
    fileID = fopen(IQ_FILE_PATH, 'r');
    if fileID == -1
        error('无法打开文件。请检查路径是否正确。');
    end
    % 读取数据为一维向量, I,Q,I,Q,...
    raw_data = fread(fileID, inf, DATA_TYPE);
    fclose(fileID);

    % 将一维向量重塑为 Nx2 的矩阵
    % -1 表示自动计算行数
    iq_pairs = reshape(raw_data, 2, [])'; % reshape 按列填充，所以先2xN再转置
    
    % 将数据转换为double类型以进行后续计算
    iq_pairs = double(iq_pairs);

    % (可选) 归一化
    if PERFORM_NORMALIZATION
        fprintf('正在执行归一化...\n');
        if strcmp(DATA_TYPE, 'int16')
            max_val = 32767;
        elseif strcmp(DATA_TYPE, 'int8')
            max_val = 127;
        else % float32
            max_val = max(abs(iq_pairs(:)));
        end
        if max_val > 0
            iq_pairs = iq_pairs / max_val;
        end
    else
        fprintf('已跳过归一化，保留原始幅度。\n');
    end
    
    % 丢弃起始样本
    if size(iq_pairs, 1) > SAMPLES_TO_DISCARD
        stable_iq_data = iq_pairs(SAMPLES_TO_DISCARD+1:end, :);
    else
        stable_iq_data = iq_pairs;
    end
    
    fprintf('数据清洗完毕。可用IQ样本数: %d\n', size(stable_iq_data, 1));

    % --- 步骤 B: 计算可切分的窗口总数 ---
    num_segments = floor((size(stable_iq_data, 1) - SEGMENT_LENGTH) / STRIDE) + 1;
    if num_segments <= 0
        error('数据不足，无法在清洗和丢弃后形成一个完整的片段。');
    end
    
    fprintf('\n步骤 2: 开始将 %d 个片段转换为 .mat 文件...\n', num_segments);

    % --- 步骤 C: 遍历、保存 .mat 文件并生成样本图 ---
    for i = 1:num_segments
        start_index = (i-1) * STRIDE + 1;
        end_index = start_index + SEGMENT_LENGTH - 1;
        
        segment_data = stable_iq_data(start_index:end_index, :); % Shape: (1024, 2)

        % --- 保存为 .mat 文件 ---
        % 首先转置为 (2, 1024)
        segment_to_save = segment_data';
        
        % 创建一个结构体来保存数据，这样变量名就是我们想要的
        data_struct.(MAT_DATASET_NAME) = segment_to_save;
        
        mat_filename = sprintf('segment_%06d.mat', i-1); % 从0开始，与Python习惯一致
        mat_save_path = fullfile(MAT_OUTPUT_DIR, mat_filename);
        
        % 保存为MATLAB v7.3格式的文件
        save(mat_save_path, '-struct', 'data_struct', '-v7.3');
        
        % --- 为前 N 个样本生成并保存分析图 ---
        if (i-1) < NUM_PLOTS_TO_SAVE
            % 调用下面的绘图函数
            save_analysis_plot_matlab(segment_to_save, i-1, PLOT_OUTPUT_DIR);
        end
        
        % 显示进度
        if mod(i, round(num_segments/10)) == 0 || i == num_segments
            fprintf('进度: %d / %d\n', i, num_segments);
        end
    end
    
    fprintf('\n转换完成！\n');
    fprintf('%d 个 .mat 文件已保存在 "%s" 文件夹中。\n', num_segments, MAT_OUTPUT_DIR);
    fprintf('%d 个分析图已保存在 "%s" 文件夹中。\n', min(num_segments, NUM_PLOTS_TO_SAVE), PLOT_OUTPUT_DIR);
    
catch ME
    fprintf('\n发生错误: %s\n', ME.message);
end

%% --- 绘图函数 (在脚本末尾定义) ---
function save_analysis_plot_matlab(iq_segment, segment_index, plot_dir)
    % 为给定的IQ片段生成并保存分析图 (MATLAB版本)
    
    % 创建一个新的、不可见的图形窗口
    fig = figure('Visible', 'off');
    
    % 准备数据 (输入形状为 2x1024)
    i_data = iq_segment(1, :);
    q_data = iq_segment(2, :);
    
    % 1. IQ时域波形
    subplot(1, 3, 1);
    plot(i_data, 'b-', 'DisplayName', 'I component');
    hold on;
    plot(q_data, 'r-', 'DisplayName', 'Q component');
    hold off;
    title('I/Q Time Domain Waveform');
    grid on; legend show;
    xlabel('Sample Index'); ylabel('Amplitude');

    % 2. 星座图
    subplot(1, 3, 2);
    scatter(i_data, q_data, 10, 'b', 'filled');
    title('Constellation Diagram');
    grid on; axis equal;
    xlabel('I (In-phase)'); ylabel('Q (Quadrature)');
    lim = max(abs(iq_segment(:))) * 1.2 + 1e-9;
    xlim([-lim, lim]); ylim([-lim, lim]);
    
    % 3. 功率谱密度 (PSD)
    subplot(1, 3, 3);
    complex_signal = i_data + 1j * q_data;
    pwelch(complex_signal, [], [], [], 1, 'centered', 'psd');
    title('Power Spectral Density (PSD)');
    
    % 设置大标题
    sgtitle(sprintf('Analysis of Segment #%d', segment_index));
    
    % 保存文件
    filename = sprintf('segment_%06d_report.png', segment_index);
    save_path = fullfile(plot_dir, filename);
    saveas(fig, save_path);
    close(fig); % 关闭图形窗口，释放内存
end