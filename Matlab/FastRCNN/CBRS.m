% 下载RPLFMSimulatedRadarDataset数据集

clear; clc; close all;

datasetZipFile = matlab.internal.examples.downloadSupportFile("SPT","data/RPLFMSimulatedRadarDataset.zip");
datasetFolder = fullfile(fileparts(datasetZipFile),"RPLFMSimulatedRadarDataset"); 
if ~exist(datasetFolder,"dir") 
    unzip(datasetZipFile,datasetFolder); 
end