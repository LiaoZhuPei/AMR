function [labelVals, labelLocsTime, labelLocsFreq] = myTFROILabeler(x, tf, t, f, tfOptions, varargin)
% MYTFROILABELER 适配 Signal Labeler 的自动标注函数

% 修复内容：修正了采样率(fs)的计算逻辑。
% 原错误：fs = 1/mean(diff(t)) 计算的是时频图的刷新率 (~7812Hz)。
% 现修正：fs = 信号点数 / 总时长，计算的是真实的物理采样率 (1MHz)。

% x 输入信号：包含时域信号数据的列向量。
% tf 频谱图功率估计值：tf为一个矩阵，其中每一列包含对应于加窗时间段的功率估计数据，
% 每一行包含对应于频率区间的功率估计数据。(能量)
% t 存储时间值：列向量，其中每个元素值代表频谱图功率估计的每个加窗时间段的中心。（时间）
% f 存储频率值：列向量，其中每个元素值代表频谱图功率估计的每个频率区间的中心。（频率）
% tfOpitions 存储有关如何计算功率估计的信息:labelSpectrogramOptions 对象
% varargin 其他输入参数:有其他输入参数，可以省略 
% 在点击"自动标注" (Auto-Label) 按钮时出现的对话框中，以有序的逗号分隔列表形式输入这些其他参数

    %% 1. 调试信息
    disp('----------------------------------------------------');
    fprintf('正在处理信号... 数据点数: %d\n', length(x));

    %% 2. 正确计算采样率
    % Signal Labeler 传入的 t 是时频图的时间轴，不是原始信号的时间轴。
    % 只有用 "原始信号点数 / 总时长" 算出来的才是真实的 fs。
    
    if length(t) > 1 && length(x) > 1
        % 计算信号总持续时间 (秒)
        duration = t(end) - t(1);
        
        % 防止 duration 为 0 的极端情况
        if duration <= 0
            fs = 1e6; 
        else
            % 采样率 = 样本总数 / 时间
            fs = length(x) / duration;
        end
    else
        fs = 1e6; % 无法计算时的默认值
    end
    
    % 取整，避免浮点误差
    fs = round(fs);
    
    fprintf('【调试】计算出的采样率 fs = %d Hz\n', fs);

    %% 3. 数据预处理
    % 强制转换为列向量，防止 stft 维度错误
    x = x(:);

    %% 4. 重新计算 STFT (保持算法一致性)
    win = hann(512);
    
    try
        % 使用计算出的正确 fs 进行 STFT
        [S, f_new, t_new] = stft(x, fs, 'FrequencyRange', 'centered', 'Window', win);
        
        % 幅度归一化 (还原真实物理幅值)
        S_norm = S / sum(win);
        Smag = abs(S_norm);
        
        fprintf('【调试】STFT计算成功。最大幅值: %.2f dB\n', max(20*log10(Smag(:)+eps)));
        
    catch ME
        warning('STFT 计算失败: %s','%s', ME.message);
        labelVals=[]; labelLocsTime=[]; labelLocsFreq=[]; return;
    end

    %% 5. 调用 ROI 检测算法
    try
        % 使用设定的参数
        % 注意：MinBandwidth = 100000 (100kHz)
        ROITable = helperFindSpecROI(Smag, f_new, t_new, ...
            'Method', 'Adaptthresh', ...
            'MinDuration', 0.005, ...     
            'MinBandwidth', 100000, ...   
            'MinArea', 300);
            
        fprintf('【调试】检测到 %d 个目标区域。\n', height(ROITable));
        
    catch ME
        warning('ROI 检测算法出错: %s','%s', ME.message);
        labelVals=[]; labelLocsTime=[]; labelLocsFreq=[]; return;
    end

    %% 6. 格式转换 (适配 Signal Labeler)
    
    if isempty(ROITable)
        % 没找到目标，返回空
        labelVals = [];
        labelLocsTime = [];
        labelLocsFreq = [];
        disp('【结果】本次未标注任何目标。');
        return;
    end
    
    % A. 生成标签名称
    % 假设所有检测到的目标都标记为 "Interference"
    numROIs = height(ROITable);
    labelVals = repmat(categorical({'Interference'}), numROIs, 1);
    
    % B. 提取坐标框 [StartT, StartF, Duration, Bandwidth]
    bbox = ROITable.BoundingBoxTF;
    
    % C. 转换为 App 要求的格式
    % 时间范围: [开始时间, 结束时间]
    labelLocsTime = [bbox(:, 1), bbox(:, 1) + bbox(:, 3)];
    
    % 频率范围: [最低频率, 最高频率]
    labelLocsFreq = [bbox(:, 2), bbox(:, 2) + bbox(:, 4)];
    
    disp('【结果】标注成功！');
    disp('----------------------------------------------------');

end