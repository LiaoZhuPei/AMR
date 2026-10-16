clear; clc; close all;

modelFolder = "E:\M_Project\FasteRCNNdata\UAV";
load(fullfile(modelFolder,"UAVDetectionNetwork","trainedUAVDetectorYoloX2025-12-04-15-34-51.mat")); 

% detector = yoloxObjectDetector(UAVdetector);
detector = UAVdetector;

disp(detector)


%% 1. 加载数据
matFileName = 'E:\M_Project\FasteRCNNdata\Tec\20170515_PartG_4.mat'; % 替换为你的文件名
if ~exist(matFileName, 'file')
    error('找不到文件: %s', matFileName);
end
load(matFileName);

% ---【请在这里手动指定你的变量名】---
% 如果你的变量名是 data，就写 x = data;
% 如果变量名是 signal，就写 x = signal;
x = signal; 

% 【关键修复】将 complex single 转为 complex double
% 这一步必须保留，否则 pspectrum 函数会报错
x = double(x);

% 假设采样率
fs = 1000000; 

%% 2. 设置参数
options = struct();
options.Leakage = 0.5;
options.UseDecibels = true;
options.MinimumThresholdMode = "auto";
options.MinimumThreshold = []; 

% 目标格式
mapFormat = "image";
mapSize = [256, 256]; 
channel = 1;

%% 3. 执行核心转化流程

% A. 获取信号
% 确保是列向量
if ismatrix(x) && size(x,2) > 1 && size(x,1) == 1, x = x'; end
% 如果是多通道矩阵，取第1列
if ismatrix(x) && size(x,2) > 1, x = x(:, channel); end

% B. 计算频谱 (pspectrum)
% complex double 输入在这里完全合法
[pxx, f, t] = pspectrum(x, fs, 'spectrogram', 'Leakage', options.Leakage);
pxx = double(pxx);

% C. 转化为 dB
if options.UseDecibels
    pxx = 10 * log10(pxx + eps);
end

% 处理阈值
minThreshold = [];
if options.MinimumThresholdMode == "specify"
    minThreshold = options.MinimumThreshold;
elseif options.MinimumThresholdMode == "auto"
    minThreshold = min(pxx(:));
end

% D. 【核心】调用官方源码版 formatForImage
imgData = local_formatForImage(pxx, mapFormat, mapSize, minThreshold, options.UseDecibels);

%% 4. 显示结果
figure;
imagesc(imgData);
axis on;
title('最终复刻版 (手动指定变量)');
xlabel('时间轴 (Time bins)');
ylabel('频率轴 (Frequency bins)');

%% ============================================================
%  【本地函数】官方源码版 formatForImage
%   保持不变
%  ============================================================
function map = local_formatForImage(map,mapFormat,mapSize,minThreshold,useDecibels)
    if useDecibels
        minValue = minThreshold;
    else
        if isempty(minThreshold)
             minValue = min(map(:));
        else
             minValue = 10^(minThreshold/10);
        end
    end
    maxValue = max(map(:));
    
    if ~isempty(mapSize)
        map = imresize(map,[mapSize(1),mapSize(2)],"nearest");
    end
    
    if mapFormat == "image"
        nColors = double(intmax("uint8")) + 1; 
        cmm = parula(nColors);
        cmm = cmm .* (nColors - 1); 
    
        denom = maxValue - minValue;
        if denom == 0, denom = 1; end
        
        m = (nColors - 1) / denom;
        b = 1 - minValue * m;
    
        normalizedMap = m .* map + b;
        height = size(map,1);
        width = size(map,2);
        idx = ceil(normalizedMap);
        
        idx(idx > nColors) = nColors;
        idx(idx < 1) = 1; 
    
        map = zeros(height, width, 3, 'uint8');
        map(:,:,1) = floor(reshape(cmm(idx, 1),[height, width]));
        map(:,:,2) = floor(reshape(cmm(idx, 2),[height, width]));
        map(:,:,3) = floor(reshape(cmm(idx, 3),[height, width]));
    end
end