function [labelVals, labelLocsTime, labelLocsFreq] = test006D(x, tf, t, f, tfOptions, varargin)
% MYTFROILABELER 适配 Signal Labeler 时频 ROI 的自定义函数
% 
% App 要求:
% [labelVals, labelLocsTime, labelLocsFreq] = fx(x, tf, t, f, tfOptions, varargin)

    %% 1. 准备参数
    % 根据时间向量重新推算采样率
    if length(t) > 1
        fs = 1 / mean(diff(t));
    else
        fs = 1e6;
    end

    %% 2. 重新计算 STFT
    
    win = hann(512);
    % x 需要确保是列向量或行向量适配
    [S, f_new, t_new] = stft(x, fs, 'FrequencyRange', 'centered', 'Window', win);
    
    % 幅度归一化
    S_norm = S / sum(win);
    Smag = abs(S_norm);

    %%ROI 检测算法
    %
    ROITable = helperFindSpecROI(Smag, f_new, t_new, ...
        'Method', 'Adaptthresh', ...
        'MinDuration', 0.005, ...     
        'MinBandwidth', 100000, ...   
        'MinArea', 300);              

    %% 4. 转换输出格式
    
    if isempty(ROITable)
        % 如果没检测到，返回空
        labelVals = [];
        labelLocsTime = [];
        labelLocsFreq = [];
        return;
    end
    
    % labelVals (类别名称)
    % categorical 类型(分类)
    numROIs = height(ROITable);
    labelVals = repmat(categorical({'A'}), numROIs, 1);
    
    %位置矩阵
    % helperFindSpecROI 的 BoundingBoxTF 格式是: [StartT, StartF, Duration, Bandwidth]
    bbox = ROITable.BoundingBoxTF;
    
    % Signal Labeler 要求 labelLocsTime 为 [StartTime, EndTime]
    labelLocsTime = [bbox(:, 1), bbox(:, 1) + bbox(:, 3)];
    
    % Signal Labeler 要求 labelLocsFreq 为 [MinFreq, MaxFreq]
    labelLocsFreq = [bbox(:, 2), bbox(:, 2) + bbox(:, 4)];

end