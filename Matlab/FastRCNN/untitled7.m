clear; clc; close all;
filename = "E:\M_Project\FasteRCNNdata\RFFrameDate\SpectrogramRFFrameDetectionData\frame_138769090412766230_bw_25E+6.mat" ;
filepath = fullfile(filename);
file = load(filepath);
sig = file.complex_array';
fs = 25e6;
win = hann(256);
figure(Position=[0 0 1000 300])
[S, f, t] = stft(sig, fs, FrequencyRange= "centered" , Window=win);
Smag = abs(S);
stft(sig, fs, FrequencyRange= "centered" , Window=win)
set(gca, YDir= "reverse" )
ROITableThresh = helperFindSpecROI(Smag,f,t,Method= "Adaptthresh" );
figure(Position=[0 0 1000 300])
helperPlotROIs(Smag,f,t,ROITableThresh)
disp(ROITableThresh)