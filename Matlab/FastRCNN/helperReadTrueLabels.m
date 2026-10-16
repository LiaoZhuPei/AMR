function outputTable = helperReadTrueLabels(filepath)
% HELPERREADTRUELABELS - Process and map true labels from signal file to STFT domain.
%   OUTPUTTABLE = HELPERREADTRUELABELS(FILEPATH) reads a .mat file containing
%   the signal, parses the filename to locate the corresponding label file,
%   and converts the labels into the Short-Time Fourier Transform (STFT) domain.
%
%   Inputs:
%       FILEPATH - Path to the .mat file containing the signal data.
%
%   Outputs:
%       OUTPUTTABLE - Table with processed labels, including bounding boxes
%                     and associated metadata.
%
%   Example:
%       outputTable = helperReadTrueLabels("data/frame123.mat");
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.

    arguments
        filepath string
    end

    % Load the signal data from the .mat file
    file = load(filepath);
    sig = file.complex_array';
    fs = 2.5e7; % Sampling frequency

    % Perform STFT
    win = hanning(256); % Hanning window
    [S, f, t] = stft(sig, fs, FrequencyRange="centered", Window=win);
    Smag = abs(S);

    % Parse the filename to generate the label file path
    newFilepath = strrep(filepath, "frame", "labels"); % Replace 'frame' with 'labels'
    newFilepath = strrep(newFilepath, ".mat", ".csv"); % Change the file extension to .csv

    % Check if the label file exists
    if exist(newFilepath, "file")
        % Read the label file if it exists
        labelsTable = readtable(newFilepath);
    else
        error("Label file does not exist: %s", newFilepath);
    end

    % Parameters for signal processing
    siglen = length(sig);       % Signal length
    winlen = 256;               % Window length
    noverlap = winlen * 0.75;   % 75% overlap
    nfft = 256;                 % FFT length

    % Convert labels from time-domain to STFT domain
    [H, W] = size(Smag);
    xmin = max(labelsTable.sample_position_start / siglen * size(Smag, 2), 1);  % Start time to STFT time bins
    xmax = min(labelsTable.sample_position_end / siglen * size(Smag, 2), W);    % End time to STFT time bins
    ymiddle = (labelsTable.freq_offset - min(f)) / (max(f) - min(f)) * length(f); % Frequency center to bins
    ybandwidth = labelsTable.bandwidth / (max(f) - min(f)) * length(f);          % Bandwidth to bins
    ymax = min(length(f) - (ymiddle - ybandwidth / 2), H); % Max frequency bin
    ymin = max(length(f) - (ymiddle + ybandwidth / 2), 1); % Min frequency bin

    % Create bounding boxes (xmin, ymin, width, height)
    blobs = [xmin, ymin, xmax - xmin, ymax - ymin];

    % Generate corresponding labels
    labels = cellfun(@(x, y) [x, y], labelsTable.class, labelsTable.rf_std, "UniformOutput", false);

    % Create output table
    outputTable = table((1:size(blobs, 1))', blobs(:, 3) .* blobs(:, 4), ...
                        [blobs(:, 1), blobs(:, 2), blobs(:, 3), blobs(:, 4)], labels, ...
                        VariableNames=["Index", "Area", "BoundingBox", "Class"]);

    % Remove rows with 'collision' class
    rowsToRemove = strcmp(outputTable.Class, "collision");
    outputTable(rowsToRemove, :) = [];

    % Map bounding boxes to time-frequency grid
    outputTable.BoundingBoxTF = helperXYMap(outputTable.BoundingBox, t, f);
end




