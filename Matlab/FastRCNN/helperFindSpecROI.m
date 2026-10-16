function out = helperFindSpecROI(Smag,f,t,Nvargs)
% HELPERFINDSPECROI - Identify and extract regions of interest (ROIs) in a spectrogram.
%   OUT = HELPERFINDSPECROI(SMAG, F, T, NVARGS) processes the input spectrogram
%   magnitude (SMAG) and returns detected ROIs based on user-defined or default
%   thresholding and segmentation methods. The function supports adaptive
%   thresholding and SAM-based segmentation methods.
%   Inputs:
%       SMAG  - Spectrogram magnitude matrix (frequency x time).
%       F     - Frequency vector.
%       T     - Time vector.
%       NVARGS.Method        - Segmentation method: 'Adaptthresh' (default) or 'SAM'.
%       NVARGS.MaxThreshold  - Maximum threshold value for ROI detection (default: 99th percentile).
%       NVARGS.MinThreshold  - Minimum threshold value for ROI detection (default: 1st percentile).
%       NVARGS.FreqRange     - Frequency range of interest as [minFreq, maxFreq] (default: all frequencies).
%       NVARGS.MinBandwidth  - Minimum frequency bandwidth of ROI (default: 1).
%       NVARGS.MinDuration   - Minimum duration (time) of ROI (default: 1).
%       NVARGS.MinArea       - Minimum area of detected ROIs (default: 0).
%       NVARGS.MaxArea       - Maximum area of detected ROIs (default: Inf).
%
%   Outputs:
%       OUT - A table containing properties of detected ROIs, including
%             bounding boxes and mapped positions to the time-frequency grid.
%
%   Example:
%       [S,f,t] = stft(sig,fs,FrequencyRange="centered",Window=win);
%       Smag = abs(S);
%       out = helperFindSpecROI(Smag, f, t, Method="SAM");
%
%   See also IMBINARIZE, BWAREAOPEN, REGIONPROPS, STREL.
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.

arguments
    Smag
    f
    t
    Nvargs.Method {mustBeMember(Nvargs.Method,{'Adaptthresh','SAM'})} = 'Adaptthresh'
    Nvargs.MaxThreshold = []
    Nvargs.MinThreshold = []
    Nvargs.FreqRange = []
    Nvargs.MinBandwidth = 1
    Nvargs.MinDuration = 1
    Nvargs.MinArea  = 0
    Nvargs.MaxArea  = Inf
end
%% Set Parameters
% Set thresholding to prevent extreme outliers from affecting the overall segmentation performance
if isempty(Nvargs.MaxThreshold)
    thresmax = prctile(Smag(:),99);
else
    thresmax = Nvargs.MaxThreshold;
end
if isempty(Nvargs.MinThreshold)
    thresmin = prctile(Smag(:),1);
else
    thresmin = Nvargs.MinThreshold;
end

% Set the target frequency range to only identify areas within these specified ranges
if isempty(Nvargs.FreqRange)
    freqidx = ones(size(f));
else
    Nvargs.FreqRange = cast(Nvargs.FreqRange,"like",f);
    freqidx = sigrangebinmask(f,Nvargs.FreqRange);
end

%% Frequency Selection and Thresholding
Sthres = Smag;
Sthres = Sthres .* freqidx; % Apply frequency range mask
Sthres(Sthres > thresmax) = thresmax; % Cap at maximum threshold
Sthres(Sthres < thresmin) = 0;       % Zero out below minimum threshold


%% Binarization and Segmentation
switch Nvargs.Method
    case "Adaptthresh"
        % Apply adaptive thresholding and morphological operations
        se = strel('rectangle',[3,3]);
        Smorph= imdilate(Sthres,se);
        Sbin = imbinarize(Smorph, "adaptive", Sensitivity=0.05);
        Sbinary =  bwareaopen(Sbin,150);
        % Find ROIs
        [labeledImage, numBlobs] = bwlabel(Sbinary);
        blobProperties = regionprops("table", labeledImage, "BoundingBox", "Area");
    case "SAM"
        % Apply SAM segmentation
        se = strel("rectangle",[3,3]);
        Smorph= imdilate(Sthres,se);
        L = helperSegmentImageSAM(Smorph);
        blobProperties = sortrows(regionprops("table", L, ["BoundingBox","Area"]),"BoundingBox");
        blobProperties =blobProperties(blobProperties.Area>0,"BoundingBox"); %原版
        %blobProperties = blobProperties(blobProperties.Area>0, ["BoundingBox", "Area"]);
        numBlobs = height(blobProperties);
end

% Filter ROIs by size and map positions
x = t;
y = f;
selectedIdx = 1:numBlobs;

BlobPropertiesSelected = blobProperties(selectedIdx,:);
idxColumn = table((1:height(BlobPropertiesSelected))', VariableNames="Index");
BlobPropertiesSelected = [idxColumn BlobPropertiesSelected];
drawPosition = helperXYMap(BlobPropertiesSelected{:,"BoundingBox"},x,y);
BlobPropertiesSelected{:,"BoundingBoxTF"} = drawPosition; %原版
%BlobPropertiesSelected.BoundingBoxTF = num2cell(drawPosition, 2);

%原版
keepIdx = BlobPropertiesSelected.Area >= Nvargs.MinArea & ...
          BlobPropertiesSelected.BoundingBoxTF(:,3) >= Nvargs.MinDuration & ...
          BlobPropertiesSelected.BoundingBoxTF(:,4) >= Nvargs.MinBandwidth;

% keepIdx = BlobPropertiesSelected.Area >= Nvargs.MinArea & ...
%           drawPosition(:,3) >= Nvargs.MinDuration & ...
%           drawPosition(:,4) >= Nvargs.MinBandwidth;

% Return results
% out = BlobPropertiesSelected;
out = BlobPropertiesSelected(keepIdx, :);

end