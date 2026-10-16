clear; clc; close all;

datafolder = "E:\M_Project\FasteRCNNdata\Tec" ; 
filename = "20170515_PartG_141.mat" ; 

filepath = fullfile(datafolder, filename); 
file = load(filepath); 
sig = file.signal'; 
fs = 1e6;

MyTrainingData = test006C(sig, fs, "UU");

disp(MyTrainingData)