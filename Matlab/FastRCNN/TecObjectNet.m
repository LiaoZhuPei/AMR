clear; clc; close all;

% 定义yolox检测器的参数
classes = {'dajp'}; % 定义类别
networkName = "tiny-coco"; 

UAVdetector = yoloxObjectDetector(networkName,classes,InputSize=[256 256 3]);

disp(UAVdetector);

rffnet = load("./RFFrameDate/SpectrogramRFFrameDetectionNetwork/networks.mat");

uavnet = rffnet.frameSegmenter;

saveDir = "./UAV/UAVDetectionNetwork";

if ~isfolder(saveDir) ; mkdir(saveDir) ; end


save(fullfile(saveDir,"UAVNetwork.mat"),"UAVdetector","uavnet");