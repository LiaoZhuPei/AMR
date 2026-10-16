clear; clc; close all;

%{
Fs = 14000;
t = (0:1/Fs:2)';
st = 0.01;
gaussFun = @(A,x,mu,f) exp(-(x-mu).^2/(2*st^2)).*sin(2*pi*f.*x)*A';
atomTimeCenters = [0.2 0.5 1 1.75];
atomFreqCenters = [2 6 2 5]*1000;
s = gaussFun([1 1 1 1]/10,t,atomTimeCenters,atomFreqCenters);
x = vco(chirp(t+.1,0,t(end),3).*exp(-2*(t-1).^2),[0.1 0.4]*Fs,Fs);
s = s/10+x;
%}

baseDir = "E:\M_Project\FasteRCNNdata";
baseDataDir = fullfile(baseDir, "Tec");

load(fullfile(baseDataDir, "20170515_PartG_10.mat"));

Fs = 1e6;
%pspectrum(s,Fs,"spectrogram", ...
%    Leakage=bt,TimeResolution=tr,OverlapPercent=op)

%atomTimes = atomTimeCenters'+[-st st]*5.5;
%atomFreqs = atomFreqCenters'+[-1 1]*200;

opts = labelSpectrogramOptions('windowlength', ...
    Window="hann",...
    Overlap=75, ...
    WindowLengthMode="specify", ...
    WindowLength=512, ...
    MinimumThresholdMode="specify");
opts.MinimumThreshold = -10;

lblDef = signalLabelDefinition("Atom", ...
    LabelDataType="logical", ...
    LabelType="roiTimeFrequency",TimeFrequencyOptions=opts);


lss = labeledSignalSet(signal,lblDef,SampleRate=Fs);

%setLabelValue(lss,1,"Atom",atomTimes,atomFreqs,true(1,4))

% imSize = [256 256];
imSize = [512 512];

%{
[sds,ads] = createDatastores(lss,"Atom", ...
    TimeFrequencyMapFormat="image", ...
    TimeFrequencyImageSize=imSize, ...
    TimeFrequencyLabelFormat="mask", ...
    TimeFrequencyMaskPriority=true);
%}

[sds, ~] = createDatastores(lss, "Atom",...
    TimeFrequencyMapFormat="image", ...
    TimeFrequencyImageSize=[imSize(1),imSize(2)], ...
    TimeFrequencyLabelFormat="xywh", ...
    TimeFrequencyIncludePartialBins=true);

image_spe = read(sds);

imagesc(image_spe)
%imagesc(read(sds))