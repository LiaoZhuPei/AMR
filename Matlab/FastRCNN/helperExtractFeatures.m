function updatedTable = helperExtractFeatures(roiTable, Nvargs)
% HELPEREXTRACTFEATURES - Extract various features from ROI table and time-frequency map.
%   UPDATEDTABLE = HELPEREXTRACTFEATURES(ROITABLE, NVARGS) computes specified
%   features for regions of interest (ROIs) in a time-frequency map (TFMap). It
%   supports features such as duration, bandwidth, central frequency, and statistical
%   moments like centroid, spread, skewness, kurtosis, and entropy.
%
%   Inputs:
%       ROITABLE - Table containing ROI information including bounding boxes.
%       NVARGS   - Name-Value pairs to specify feature extraction options.
%                  Options include:
%                     TFMap (required)  - Time-frequency magnitude map.
%                     Signal            - Original signal corresponding to the ROIs.
%                     FreqVec           - Frequency vector for the TFMap.
%                     Duration          - Extract ROI duration (default: false).
%                     Bandwidth         - Extract ROI bandwidth (default: false).
%                     CentralFrequency  - Extract ROI central frequency (default: false).
%                     Spectrum          - Compute ROI spectrum (default: false).
%                     MedianMag         - Compute median magnitude for each ROI (default: false).
%                     Top25Mag          - Compute top 25th percentile magnitude (default: false).
%                     MeanMag           - Compute mean magnitude for each ROI (default: false).
%                     Centroid          - Compute spectral centroid (default: false).
%                     Spread            - Compute spectral spread (default: false).
%                     Skewness          - Compute spectral skewness (default: false).
%                     Kurtosis          - Compute spectral kurtosis (default: false).
%                     Entropy           - Compute spectral entropy (default: false).
%                     Verbose           - Print progress messages (default: false).
%
%   Outputs:
%       UPDATEDTABLE - Updated table with the extracted features as new columns.
%
%   Example:
%       updatedTable = helperExtractFeatures(roiTable, ...
%                       TFMap=Smag, FreqVec=f, Duration=true, Centroid=true);
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.


arguments
    roiTable
    Nvargs.TFMap = []
    Nvargs.Signal = []
    Nvargs.FreqVec = []
    Nvargs.Duration logical = false
    Nvargs.Bandwidth logical = false
    Nvargs.CentralFrequency logical = false
    Nvargs.Spectrum logical = false
    Nvargs.MedianMag logical = false
    Nvargs.Top25Mag logical = false
    Nvargs.MeanMag logical = false

    % Moments
    Nvargs.Centroid  logical = false
    Nvargs.Spread logical = false
    Nvargs.Skewness logical = false
    Nvargs.Kurtosis logical = false
    Nvargs.Entropy logical = false
    Nvargs.Verbose logical = false
end
% Initialize the updated table with the input table
updatedTable = roiTable;
Smag = Nvargs.TFMap;
[H,W] = size(Smag);
Nroi = height(roiTable);


%% Duration
feature = 'Duration';
if Nvargs.(feature)
    if  ~ismember(feature, updatedTable.Properties.VariableNames)
        featureVal = roiTable.BoundingBoxTF(:,3);
        updatedTable.(feature) = featureVal;
    end
end

%% CentralFrequency
feature = 'CentralFrequency';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        featureVal = roiTable.BoundingBoxTF(:,2)+roiTable.BoundingBoxTF(:,4)/2;
        updatedTable.(feature) = featureVal;
    end
end
%%

%% Bandwidth
feature = 'Bandwidth';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        featureVal = roiTable.BoundingBoxTF(:,4);
        updatedTable.(feature) = featureVal;
    end
end

%% Magnitude
feature = 'Top25Mag';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)

        featureVal = zeros(Nroi,1);
        for i = 1:Nroi
            % Sroi in Smag position
            boundary = roiTable{i,"BoundingBox"};
            xmin = int32(max(1,(boundary(1))));
            ymin = int32(max(1,(boundary(2))));
            xmax = int32(min(W,round(boundary(1)+boundary(3))));
            ymax = int32(min(H,round(boundary(2)+boundary(4))));
            Sroi = Smag(ymin:ymax,xmin:xmax);
            featureVal(i,:) = prctile(Sroi(:),25);
        end

        updatedTable.(feature) = featureVal;
        % disp(['Finish ',feature])
    end
end

feature = 'MedianMag';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)

        featureVal = zeros(Nroi,1);
        for i = 1:Nroi
            % Sroi in Smag position
            boundary = roiTable{i,"BoundingBox"};
            xmin = int32(max(1,(boundary(1))));
            ymin = int32(max(1,(boundary(2))));
            xmax = int32(min(W,round(boundary(1)+boundary(3))));
            ymax = int32(min(H,round(boundary(2)+boundary(4))));
            Sroi = Smag(ymin:ymax,xmin:xmax);
            featureVal(i,:) = prctile(Sroi(:),50);
        end

        updatedTable.(feature) = featureVal;
    end
end

feature = 'MeanMag';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)

        featureVal = zeros(Nroi,1);
        for i = 1:Nroi
            % Sroi in Smag position
            boundary = roiTable{i,"BoundingBox"};
            xmin = int32(max(1,(boundary(1))));
            ymin = int32(max(1,(boundary(2))));
            xmax = int32(min(W,round(boundary(1)+boundary(3))));
            ymax = int32(min(H,round(boundary(2)+boundary(4))));
            Sroi = Smag(ymin:ymax,xmin:xmax);
            featureVal(i,:) = mean(Sroi(:));
        end

        updatedTable.(feature) = featureVal;
    end
end

%% Spectrum
feature = 'Spectrum';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        updatedTable = computeSpectrum(Smag,updatedTable);
    end
end

%% Centroid
feature = 'Centroid';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        updatedTable = computeCentroid(Smag,updatedTable,Nvargs);
        % disp(['Finish ',feature])
    end
end

%% Spread
feature = 'Spread';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        updatedTable = computeSpread(Smag,updatedTable,Nvargs);
        % disp(['Finish ',feature])
    end
end


%% Skewness
feature = 'Skewness';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        updatedTable = computeSkewness(Smag,updatedTable,Nvargs);
    end
end

%% Entropy
feature = 'Entropy';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        updatedTable = computeEntropy(Smag,updatedTable,Nvargs);
    end
end

%% Kurtosis
feature = 'Kurtosis';
if Nvargs.(feature)
    if ~ismember(feature, updatedTable.Properties.VariableNames)
        updatedTable = computeKurtosis(Smag,updatedTable,Nvargs);
    end
end
end

% Kurtosis
function updatedTable = computeKurtosis(Smag,updatedTable,Nvargs)
    feature = "Kurtosis";
    Nroi = height(updatedTable);
    featureVal = zeros(Nroi,1);

    if ~ismember("Spread", updatedTable.Properties.VariableNames)
        updatedTable = computeSpread(Smag,updatedTable);
    end

    spectrumMat = updatedTable.Spectrum; % sk
    centroidMat = updatedTable.Centroid; % mu1
    spreadMat = updatedTable.Spread; % mu2
    
    for i = 1:Nroi
        % The spectral kurtosis measures the flatness, or non-Gaussianity,
        % of the spectrum around its centroid. Conversely, it is used to
        % indicate the peakiness of a spectrum. For example, as the white
        % noise is increased on the speech signal, the kurtosis decreases,
        % indicating a less peaky spectrum.
        featureVal(i,:) = sum( (Nvargs.FreqVec(:)-centroidMat(i)).^4.*spectrumMat(i,:)'  ) / spreadMat(i).^4*sum(spectrumMat(i,:));
    end
    updatedTable.(feature) = featureVal;
end

% Entropy
function updatedTable = computeEntropy(Smag,updatedTable,~)
    feature = "Entropy";
    Nroi = height(updatedTable);
    featureVal = zeros(Nroi,1);

    if ~ismember("Spectrum", updatedTable.Properties.VariableNames)
        updatedTable = computeSpectrum(Smag,updatedTable);
    end

    spectrumMat = updatedTable.Spectrum; % sk
    
    for i = 1:Nroi
        % Spectral entropy has been used successfully in voiced/unvoiced
        % decisions for automatic speech recognition [6]. Because entropy
        % is a measure of disorder, regions of voiced speech have lower
        % entropy compared to regions of unvoiced speech.
        featureVal(i,:) = -sum( spectrumMat(i,:).*log(spectrumMat(i,:)+eps)  ) / log(sum(spectrumMat(i,:)>0));
    end
    updatedTable.(feature) = featureVal;
end

function updatedTable = computeCentroid(Smag,updatedTable,Nvargs)
    feature = "Centroid";
    Nroi = height(updatedTable);
    featureVal = zeros(Nroi,1);
    % [H,W] = size(Smag);

    if ~ismember("Spectrum", updatedTable.Properties.VariableNames)
        updatedTable = computeSpectrum(Smag,updatedTable);
    end
    spectrumMat = updatedTable.Spectrum;
    
    for i = 1:Nroi
        % The spectral centroid (spectralCentroid) is the frequency-weighted sum normalized by the unweighted sum 
        spec = spectrumMat(i,:);
        featureVal(i,:) = sum(Nvargs.FreqVec(:).*spec(:))/sum(spec(:));
    end
    updatedTable.(feature) = featureVal;
end

function updatedTable = computeSpread(Smag,updatedTable,Nvargs)
    feature = "Spread";
    Nroi = height(updatedTable);
    featureVal = zeros(Nroi,1);

    if ~ismember("Centroid", updatedTable.Properties.VariableNames)
        updatedTable = computeCentroid(Smag,updatedTable,Nvargs);
    end
    spectrumMat = updatedTable.Spectrum;
    centroidMat = updatedTable.Centroid;
    
    for i = 1:Nroi
        % The spectral centroid (spectralCentroid) is the frequency-weighted sum normalized by the unweighted sum 
        spec = spectrumMat(i,:);
        featureVal(i,:) = (sum((Nvargs.FreqVec(:)-centroidMat(i)).^2.*spec(:))/sum(spec(:))).^0.5;
    end
    updatedTable.(feature) = featureVal;
end

function updatedTable = computeSkewness(Smag,updatedTable,Nvargs)
    feature = "Skewness";
    Nroi = height(updatedTable);
    featureVal = zeros(Nroi,1);

    if ~ismember("Spread", updatedTable.Properties.VariableNames)
        updatedTable = computeSpread(Smag,updatedTable);
    end
    spectrumMat = updatedTable.Spectrum; % sk
    centroidMat = updatedTable.Centroid; % mu1
    spreadMat = updatedTable.Spread; % mu2

    for i = 1:Nroi
        spec = spectrumMat(i,:);
        featureVal(i,:) = sum((Nvargs.FreqVec(:)-centroidMat(i)).^3.*spec(:))/spreadMat(i).^3/sum(spec(:));
    end
    updatedTable.(feature) = featureVal;
end

function updatedTable = computeSpectrum(Smag,updatedTable)
    feature = "Spectrum";
    Nroi = height(updatedTable);
    [H,W] = size(Smag);
    featureVal = zeros(Nroi,size(Smag,1));
   
    for i = 1:Nroi
        % Sroi in Smag position
        boundary = updatedTable{i,"BoundingBox"};
    
        xmin = int32(max(1,(boundary(1))));
        ymin = int32(max(1,(boundary(2))));
        xmax = int32(min(W,round(boundary(1)+boundary(3))));
        ymax = int32(min(H,round(boundary(2)+boundary(4))));
    
        Sroi = Smag(ymin:ymax,xmin:xmax);
        magvec = zeros(size(Smag,1),1);
        magvec(ymin:ymax,1) = mean(Sroi,2);
        magvec = magvec./max(magvec);
        featureVal(i,:) = magvec';
    end

    updatedTable.(feature) = featureVal;
end