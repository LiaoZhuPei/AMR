clear; clc; close all;

% Define the location the dataset will be stored. By default, the temporary
% folder for the system will be used.
baseDir = "E:\M_Project\FasteRCNNdata";
baseDataDir = fullfile(baseDir, "Tec");
baseNetworkDir = fullfile(baseDir, "UAV\UAVDetectionNetwork");

% 加载标签信息文件
load(fullfile(baseDir, "UAVlss.mat"));

% 信号->频谱图->图像
imageSize = [256 256];

%{
labelDefinitionName = "UAV";
[odSds,odLds] = ls.createDatastores(labelDefinitionName, ...
    TimeFrequencyLabelFormat="xywh",...
    TimeFrequencyMapFormat="image",...
    TimeFrequencyImageSize=[imageSize(1), imageSize(2)], ...
    TimeFrequencyIncludePartialBins=true);

[specImage, specInfo] = odSds.read;
odSds.reset;
specLabel = odLds.preview;

imagesc(specImage);
%}
