clear; clc; close all;

% UAV_yolox_train_network
% 训练

% Define the location the dataset will be stored. By default, the temporary
% folder for the system will be used.
baseDir = "E:\M_Project\FasteRCNNdata";
baseDataDir = fullfile(baseDir, "Tec");
baseNetworkDir = fullfile(baseDir, "UAV\UAVDetectionNetwork");

% 加载标签信息文件
load(fullfile(baseDir, "UAVlss.mat"));

% 信号->频谱图->图像
imageSize = [256 256];

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

catsObjectDetection = "dajp";
helperObjectDetectionPlotter(specImage,specInfo,specLabel,catsObjectDetection);

% 划分数据集->训练+验证+测试
parts = [80 10 10];
s = RandStream("mt19937ar", Seed=0);
numFiles = ls.NumMembers;
shuffledIndics = randperm(s, numFiles);

numTrain = floor(numFiles * parts(1) / 100);
numVal = floor(numFiles * parts(2) / 100);

odSdsTrain = subset(odSds, shuffledIndics(1:numTrain));
odSdsVal = subset(odSds, shuffledIndics(numTrain + (1:numVal)));
odSdsTest = subset(odSds, shuffledIndics(numTrain + numVal + 1:end));

odLdsTrain = subset(odLds, shuffledIndics(1:numTrain));
odLdsVal = subset(odLds, shuffledIndics(numTrain + (1:numVal)));
odLdsTest = subset(odLds, shuffledIndics(numTrain + numVal + 1:end));

% 合并数据和标签用于训练
odCdsTrain = combine(odSdsTrain,odLdsTrain);
odCdsVal = combine(odSdsVal,odLdsVal);
odCdsTest = combine(odSdsTest,odLdsTest);

% 训练网络
trainNow = true; % 为false则使用预训练网络，为true从头开始训练

if trainNow
    UAVdetector = yoloxObjectDetector("tiny-coco",catsObjectDetection, ...
        InputSize=[imageSize(1),imageSize(2),3]);
end

options = trainingOptions("sgdm", ...
    InitialLearnRate=5e-4, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.99, ...
    LearnRateDropPeriod=1, ...
    MiniBatchSize=20, ...
    MaxEpochs=100, ...
    ExecutionEnvironment="auto", ...
    Shuffle="every-epoch", ...
    VerboseFrequency=25, ...
    ValidationFrequency=100, ...
    ValidationData=odCdsVal, ...
    ResetInputNormalization=false, ...
    OutputNetwork="best-validation-loss", ...
    GradientThreshold=30, ...
    L2Regularization=5e-4, ...
    Plots="training-progress");

if trainNow
    UAVdetector = trainYOLOXObjectDetector(odCdsTrain,UAVdetector, ...
        options,"FreezeSubNetwork","none");
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save(fullfile(baseNetworkDir,"trainedUAVDetectorYoloX"+modelDateTime),"UAVdetector")
else
    load(fullfile(baseNetworkDir,"UAVNetwork.mat"), "UAVdetector");
end

% 结果评价
detectionThresh = 0.1;
detectorResults = detect(UAVdetector,odCdsTest,...
    AutoResize=false,MiniBatchSize=1,Threshold=detectionThresh);

metricsObjectDetection = evaluateObjectDetection(detectorResults,odCdsTest);

AP = averagePrecision(metricsObjectDetection);
disp(table(catsObjectDetection',AP,VariableNames=["Category","AP"]));

[confMat,confusionClassNames] = confusionMatrix(metricsObjectDetection, ...
    scoreThreshold=detectionThresh);

figure
confusionchart(confMat{1},confusionClassNames)