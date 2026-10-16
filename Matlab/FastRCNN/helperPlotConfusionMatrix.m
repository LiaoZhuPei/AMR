function confMat = helperPlotConfusionMatrix(predictedMask, groundTruthMask)
% HELPERPLOTCONFUSIONMATRIX - Plot confusion matrix for binary object detection.
%   CONFMAT = HELPERPLOTCONFUSIONMATRIX(PREDICTEDMASK, GROUNDTRUTHMASK)
%   computes the confusion matrix for binary object detection, displaying it as
%   a confusion chart with row- and column-normalized summaries.
%
%   Inputs:
%       PREDICTEDMASK   - Binary mask predicted by the model (same size as GROUNDTRUTHMASK).
%       GROUNDTRUTHMASK - Ground truth binary mask.
%
%   Outputs:
%       CONFMAT         - 2x2 confusion matrix as [TP, FP; FN, TN].
%
%   Example:
%       confMat = helperPlotConfusionMatrix(predictedMask, groundTruthMask);
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.


    % Compute confusion matrix components
    truePositives = (predictedMask == 1) & (groundTruthMask == 1);
    falsePositives = (predictedMask == 1) & (groundTruthMask == 0);
    falseNegatives = (predictedMask == 0) & (groundTruthMask == 1);
    trueNegatives = (predictedMask == 0) & (groundTruthMask == 0);
    
    % Sum up the values to get the total counts
    tpCount = sum(truePositives, "all");
    fpCount = sum(falsePositives, "all");
    fnCount = sum(falseNegatives, "all");
    tnCount = sum(trueNegatives, "all");
    
    % Create the confusion matrix
    confMat = [tpCount, fpCount; fnCount, tnCount];
    
    % Display confusion matrix chart
    confusionchart(confMat, ["Object", "Background"], ...
                   RowSummary="row-normalized", ...
                   ColumnSummary="column-normalized");
    title("Confusion Matrix for Object Detection");
end
