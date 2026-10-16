clear; clc; close all;

% 单文件读取
baseDir = "E:\M_Project\FasteRCNNdata";
baseDataDir = fullfile(baseDir, "Tec");
load(fullfile(baseDataDir, "20170515_PartG_100.mat"));

% 设定采样率
Fs = 1e6;

% 定义时频图绘制参数（与标注器相同的参数）
% image_spe000.png
opts = labelSpectrogramOptions('windowlength', ...
    Window="hann",...
    Overlap=75, ...
    WindowLengthMode="specify", ...
    WindowLength=512, ...
    MinimumThresholdMode="specify");
opts.MinimumThreshold = -10; 

% 相同信号下不同STFT参数的时频图
% image_spe001.png Overlap=20 WindowLength=128 ; opts.MinimumThreshold = -50;
% image_spe002.png Overlap=20 WindowLength=512 ; opts.MinimumThreshold = -50;
% image_spe003.png Overlap=20 WindowLength=1024 ; opts.MinimumThreshold = -50;
% image_spe004.png Overlap=20 WindowLength=64 ; opts.MinimumThreshold = -50;
% image_spe005.png Overlap=35 WindowLength=64 ; opts.MinimumThreshold = -50;
% image_spe006.png Overlap=55 WindowLength=512 ; opts.MinimumThreshold = -50;
% image_spe007.png Overlap=55 WindowLength=1024 ; opts.MinimumThreshold = -50;
% image_spe008.png Overlap=55 WindowLength=1024 ; opts.MinimumThreshold = -80;
% image_spe009.png Overlap=25 WindowLength=64 ; opts.MinimumThreshold = -80;

% 用于修改STFT参数以测试模型的稳定性
% opts = labelSpectrogramOptions('windowlength', ...
%     Window="hann",...
%     Overlap=55, ...
%     WindowLengthMode="specify", ...
%     WindowLength=512, ...
%     MinimumThresholdMode="specify");
% opts.MinimumThreshold = -80; 

% 制作伪标签（适应模型）
lblDef = signalLabelDefinition("Atom", ...
    LabelDataType="logical", ...
    LabelType="roiTimeFrequency", ...
    TimeFrequencyOptions=opts);
lss = labeledSignalSet(signal,lblDef,SampleRate=Fs);

% 图像大小
imSize = [256 256];
% imSize = [512 512];

% 将时频图转化为图像
[sds, ~] = createDatastores(lss, "Atom",...
    TimeFrequencyMapFormat="image", ...
    TimeFrequencyImageSize=[imSize(1),imSize(2)], ...
    TimeFrequencyLabelFormat="xywh", ...
    TimeFrequencyIncludePartialBins=true);

image_spe = read(sds); %image_spe为图像
%imwrite(image_spe, 'image_spe009.png');

imagesc(image_spe)

%加载模型
modelFolder = "E:\M_Project\FasteRCNNdata\UAV";
load(fullfile(modelFolder,"UAVDetectionNetwork","trainedUAVDetectorYoloX2025-12-04-15-34-51.mat")); 

%定义检测器（用于推理）
detector = UAVdetector;

disp(detector)

%% 信号-绘制时频图-转为图片-推理
% [bboxes,scores,labels] = detect(detector,image_spe);
% 
% detectedImg = insertObjectAnnotation(image_spe,"Rectangle",bboxes,labels);
% figure
% imshow(detectedImg)


%% 调用实际图片验证是否会误标注 
%调用图片-推理
%选择图片
% I = imread("figure3.png");  % 图片上可能不标注
% I = imread("figure4.png");  % 图片上可能误标注
I = imread("image_spe000.png"); % 相同信号不同STFT参数的标注情况

% 运行
[bboxes,scores,labels] = detect(detector,I);
detectedImg = insertObjectAnnotation(I,"Rectangle",bboxes,labels);
figure
imshow(detectedImg)
