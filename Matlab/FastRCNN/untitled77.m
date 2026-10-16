clear; clc; close all;

% Define the location the dataset will be stored. By default, the temporary
% folder for the system will be used.
baseDir = "E:\M_Project\FasteRCNNdata\RFFrameDate";
baseDataDir = fullfile(baseDir, "SpectrogramRFFrameDetectionData");
baseNetworkDir = fullfile(baseDir, "SpectrogramRFFrameDetectionNetwork");

load(fullfile(baseDataDir, "preLabeledSignalSet.mat"));

% 修补旧的路径，此处数据集SpectrogramRFFrameDetectionData是引用别人的，
% 因此保存的信息的路径是别人的'M:\SpectrogramRFFrameDetectionData'，
% 这里需要指定其中某个成员的路径为我们自己设置的路径baseDataDir
labeledLss.setAlternateFileSystemRoots([labeledLss.Source.Folders, baseDataDir]); 

imageSize = [256 256];

labelDefinitionName = "RFFrame";
[odSds,odLds] = labeledLss.createDatastores(labelDefinitionName, ...
    TimeFrequencyLabelFormat="xywh", ...
    TimeFrequencyMapFormat="image", ...
    TimeFrequencyImageSize=[imageSize(1),imageSize(2)], ...
    TimeFrequencyIncludePartialBins=true);

[specImage, specInfo] = odSds.read;
odSds.reset;
specLabel = odLds.preview;

imagesc(specImage);