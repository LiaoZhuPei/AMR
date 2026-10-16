function mask = helperBoundingBoxesToMask(boundingBoxes, W, H)
% HELPERBOUNDINGBOXESTOMASK - Convert bounding boxes to a binary mask.
%   MASK = HELPERBOUNDINGBOXESTOMASK(BOUNDINGBOXES, W, H) generates a binary
%   mask of size HxW, where pixels within the bounding boxes are set to 1.
%
%   Inputs:
%       BOUNDINGBOXES - Nx4 matrix, where each row specifies a bounding box
%                       as [x, y, width, height].
%       W            - Width of the output mask.
%       H            - Height of the output mask.
%
%   Outputs:
%       MASK         - Binary mask of size HxW with 1s in bounding box regions.
%
%   Example:
%       boundingBoxes = [10, 20, 30, 40; 50, 60, 70, 80];
%       mask = helperBoundingBoxesToMask(boundingBoxes, 100, 100);
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.

    % Initialize the mask to zeros
    mask = zeros(H, W);
    
    % Loop through each bounding box and fill in the mask
    for idx = 1:size(boundingBoxes, 1)
        % Extract the bounding box
        x = boundingBoxes(idx, 1);
        y = boundingBoxes(idx, 2);
        w = boundingBoxes(idx, 3);
        h = boundingBoxes(idx, 4);
        
        % Convert bounding box coordinates to pixel indices
        xStart = max(1, round(x));
        yStart = max(1, round(y));
        xEnd = min(W, round(x + w));
        yEnd = min(H, round(y + h));
        
        % Set the corresponding area of the mask to 1
        mask(yStart:yEnd, xStart:xEnd) = 1;
    end
end