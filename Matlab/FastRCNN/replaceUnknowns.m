function [labelsOut, info] = replaceUnknowns(dataIn, info)
    % This function replaces the missing values of a categorical array with
    % the string "undefined".
    missingLabels = ismissing(dataIn{1});
    labels = dataIn{1};
    labels(missingLabels) = "undefined";
    labelsOut = {labels};
end