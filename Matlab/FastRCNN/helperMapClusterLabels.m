function predictedLabelsMapped = helperMapClusterLabels(trueLabels, predictedLabels)
% HELPERMAPCLUSTERLABELS - Map predicted cluster labels to true labels.
%   PREDICTEDLABELSMAPPED = HELPERMAPCLUSTERLABELS(TRUELABELS, PREDICTEDLABELS)
%   remaps predicted cluster labels to align with the true labels using the
%   Hungarian algorithm when the number of unique labels matches. For unequal
%   cluster counts, it maps based on the most common true label within each
%   predicted cluster.
%
%   Inputs:
%       TRUELABELS      - Ground truth labels as a vector.
%       PREDICTEDLABELS - Predicted cluster labels as a vector.
%
%   Outputs:
%       PREDICTEDLABELSMAPPED - Vector of remapped predicted labels aligned
%                               with the true labels.
%
%   Example:
%       trueLabels = [1; 1; 2; 2; 3; 3];
%       predictedLabels = [2; 2; 3; 3; 1; 1];
%       predictedLabelsMapped = helperMapClusterLabels(trueLabels, predictedLabels);
%       trueLabels = [1; 1; 2; 2; 2; 3];
%       predictedLabels = [2; 2; 3; 3; 1; 1];
%       predictedLabelsMapped = helperMapClusterLabels(trueLabels, predictedLabels);
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.

    % Extract unique labels from true and predicted clusters
    [C1, ~, ic1] = unique(trueLabels);       % Unique true labels
    [C2, ~, ic2] = unique(predictedLabels); % Unique predicted labels

    % Case 1: Equal number of unique labels (Hungarian algorithm)
    if length(C1) == length(C2)
        % Create the confusion matrix
        confusionMatrix = zeros(length(C1), length(C2));
        for i = 1:numel(ic1)
            confusionMatrix(ic1(i), ic2(i)) = confusionMatrix(ic1(i), ic2(i)) + 1;
        end

        % Use negative since Hungarian algorithm minimizes
        confusionMatrix = -confusionMatrix;

        % Solve using Munkres global nearest neighbor assignment algorithm
        [assignments, ~, ~] = assignmunkres(confusionMatrix, 100);

        % Map predicted clusters to true clusters
        mappedLabels = [C1(assignments(:,1)), C2(assignments(:,2))];
        disp(mappedLabels); % Display mappings for debugging
        predictedLabelsMapped = mappedLabels(ic2);

    % Case 2: Unequal number of unique labels (mode-based mapping)
    else
        mappedLabels = zeros(size(C2));
        for i = 1:length(C2)
            idx = ic2 == i;             % Find indices for current predicted label
            truelabel = ic1(idx);       % Get corresponding true labels
            mappedLabels(i) = mode(truelabel); % Assign most common true label
        end
        predictedLabelsMapped = C1(mappedLabels(ic2));
    end
end
