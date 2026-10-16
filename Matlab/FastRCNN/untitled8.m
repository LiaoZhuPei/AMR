clear; clc; close all;

% ================= 配置区域 =================
filename = "E:\M_Project\FasteRCNNdata\RFFrameDate\SpectrogramRFFrameDetectionData\frame_138769090412766230_bw_25E+6.mat"; 
fs = 1e6; % 采样率 1MHz
% ===========================================

% 1. 读取文件
filepath = fullfile(filename);
if ~isfile(filepath), error('找不到文件'); end
file = load(filepath);
vars = fieldnames(file);
sig = file.(vars{1}); % 自动读取变量
sig = double(sig);
if isrow(sig), sig = sig'; end

% 2. STFT 处理
win = hann(256);
[S, f, t] = stft(sig, fs, FrequencyRange="centered", Window=win);

% 3. 幅度转 dB (实数化)
Smag_Linear = abs(S); 
Smag_dB = 20*log10(Smag_Linear + eps); 

% 清洗数据，防止 Inf/NaN
Smag_dB(isinf(Smag_dB)) = min(Smag_dB(~isinf(Smag_dB)));
Smag_dB(isnan(Smag_dB)) = min(Smag_dB(:));

% 4. ROI 检测
detect_thresh = 0.35; % 阈值
try
    ROITable = helperFindSpecROI(Smag_dB, f, t, Method="Fixed", Threshold=detect_thresh);
catch
    ROITable = helperFindSpecROI(Smag_dB, f, t);
end

% =========================================================
% 5. 智能绘图 (自动适配表格列名)
% =========================================================
figure(Position=[100 100 1000 400])
imagesc(t, f, Smag_dB); 
axis xy; 
set(gca, 'YDir', 'normal'); 
colormap('jet'); colorbar;
title('Detection Results');
xlabel('Time (s)'); ylabel('Frequency (Hz)');
hold on;

if ~isempty(ROITable)
    % 获取表格的所有列名
    cols = ROITable.Properties.VariableNames;
    
    for i = 1:height(ROITable)
        % --- 自动匹配列名逻辑 ---
        if ismember('TimeLimits', cols)
            % 情况A: 官方标准输出 (TimeLimits, FrequencyLimits)
            t_vals = ROITable.TimeLimits(i, :);
            f_vals = ROITable.FrequencyLimits(i, :);
            tMin = t_vals(1); tMax = t_vals(2);
            fMin = f_vals(1); fMax = f_vals(2);
            
        elseif ismember('tMin', cols)
            % 情况B: 自定义输出 (tMin, tMax...)
            tMin = ROITable.tMin(i); tMax = ROITable.tMax(i);
            fMin = ROITable.fMin(i); fMax = ROITable.fMax(i);
            
        elseif ismember('xmin', cols)
            % 情况C: 某些版本输出 (xmin, ymin...)
            tMin = ROITable.xmin(i); tMax = ROITable.xmax(i);
            fMin = ROITable.ymin(i); fMax = ROITable.ymax(i);
            
        else
            % 如果都不是，打印列名并停止，防止报错
            warning('未知的表格列名: %s', strjoin(cols, ', '));
            break; 
        end
        % -----------------------
        
        % 绘制矩形框
        width = tMax - tMin;
        height_freq = fMax - fMin;
        rectangle('Position', [tMin, fMin, width, height_freq], ...
                  'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '-');
    end
    fprintf('成功绘制 %d 个检测框。\n', height(ROITable));
else
    fprintf('未检测到 ROI (表格为空)。\n');
end
hold off;

% 在命令行显示检测到的具体数据
disp('检测结果表内容：');
disp(ROITable);