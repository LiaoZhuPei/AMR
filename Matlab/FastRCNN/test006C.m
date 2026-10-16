function labeledTable = test006C(sig, fs, targetClassName)
%TEST006C 自动标注单一类别
%输入：
%   si  - 输入信号向量
%   fs  - 采样率
%   targetClassName    -类别名称
%输出：
%   labeledTable       -包含特征、坐标和标签

% stft计算
win = hann(512);
[S, f,t] = stft(sig, fs, 'FrequencyRange', 'centered', 'Window', win);

% 幅值归一化
S_norm = S/sum(win);
Smag = abs(S_norm);

% ROI检测
ROITable = helperFindSpecROI( Smag, f, t, ...
    'Method', 'Adaptthresh', ...
    'MinDuration', 0.005, ...
    'MinBandwidth', 100000, ...
    'MinArea', 300);

if isempty(ROITable)
    labeledTable = table();
    warning('未检测到任何信号');
    return;
end

% 特征提取
% 提取中值、幅值等特征
labeledTable = helperExtractFeatures(ROITable,'TFMap',Smag,'MedianMag',true);

% 添加标签（单类别）
numROIs = height(labeledTable); % 检测到的数量

labels = repmat(categorical(targetClassName), numROIs, 1); % 创建标签列，（repmat将单类别复制N行）

labeledTable.Label = labels; %将标签加入表格

% 清理不需要的列
% 可删除指定变量
% labeledTable = removevars(labeledTable, {'Label','Area'});

% 将label移至第一列，便于观察
labeledTable = movevars(labeledTable, 'Label', 'Before', 1);


end