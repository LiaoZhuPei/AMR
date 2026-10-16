%% 自动标注1 - 基于局部自适应灰度阈值的定位
% 将频谱图视为灰度图像。使用局部自适应阈值可以区分包含信号的区域（前景）和不包含信号的区域（背景）
%The helper function performs these steps:
%1.Preprocess the image using imdilate to smooth the spectrogram to some extent.
%2.Use adaptthresh to segment the preprocessed spectrogram.
%3.Extract the segmentation results, collect the bounding boxes, and save and organize them into a table format.

clear; clc; close all;

datafolder = "E:\M_Project\FasteRCNNdata\Tec" ; 
filename = "20170515_PartG_20.mat" ; 

filepath = fullfile(datafolder, filename); 
file = load(filepath); 
sig = file.signal'; 
fs = 1e6; 
win = hann(512); 
figure(Position=[0 0 1000 300]) 
% stft绘图模式（无参数输出）
% stft图的幅值上下限（dB）在stft函数中的displaySTFT函数中设置
stft(sig, fs, FrequencyRange= "centered" , Window=win) 

set(gca, YDir= "reverse" ) % 颠倒y轴

% stft计算模式（有参数输出）
[S, f, t] = stft(sig, fs, FrequencyRange= "centered" , Window=win); 
S_norm = S / (sum(win));  % 幅度归一化，消除窗函数和FFT带来的"增益"放大效益

% % 原始矩阵的最大 dB 值
% max_val_S = max(abs(S(:)));
% max_dB_S  = 20*log10(max_val_S);
% fprintf('原始矩阵 S 的最大值: %.2f dB\n', max_dB_S);
% % 归一化矩阵的最大 dB 值
% max_val_Norm = max(abs(S_norm(:)));
% max_dB_Norm  = 20*log10(max_val_Norm);
% fprintf('归一化矩阵 S_norm 的最大值: %.2f dB\n', max_dB_Norm);

% 计算幅值
Smag = abs(S_norm); 
Smag_dB = mag2db(Smag);
%histogram(Smag(:));

segmentNow =true;

if segmentNow 
    ROITableSAM = helperFindSpecROI(Smag,f,t,Method= "SAM" ); %#ok<UNRCH> 
else 
    file = load( "savedSAMTFLabelResults.mat" ); 
    ROITableSAM = file.ROITableSAM; end

ROITableSAMFeature = helperExtractFeatures(ROITableSAM, ... 
    TFMap=Smag, ... 
    MedianMag=1); 

threshold = 0.01;
selectedidx = ROITableSAMFeature{:, "MedianMag" } > threshold; 
ROITableSAMSelect = ROITableSAMFeature(selectedidx,:); 

figure(Position=[0 0 1000 300]) 
helperPlotROIs(Smag,f,t,ROITableSAMSelect)