function drawPosition = helperXYMap(thisBoundingBox, X, Y)
% HELPERXYMAP - Map bounding box coordinates to a specified (x, y) grid.
%   DRAWPOSITION = HELPERXYMAP(THISBOUNDINGBOX, X, Y) converts bounding box
%   coordinates from pixel indices to the corresponding (x, y) values on a
%   scaled grid defined by the input vectors X and Y.
%
%   Inputs:
%       THISBOUNDINGBOX - Nx4 matrix where each row represents a bounding box
%                         as [x, y, width, height] in pixel indices.
%       X               - Vector representing the x-coordinates of the grid.
%       Y               - Vector representing the y-coordinates of the grid.
%
%   Outputs:
%       DRAWPOSITION    - Nx4 matrix where each row represents a bounding box
%                         as [x, y, scaledWidth, scaledHeight] in the (x, y) grid.
%
%   Example:
%       thisBoundingBox = [10, 20, 30, 40];
%       x = linspace(0, 100, 100);
%       y = linspace(0, 50, 50);
%       drawPosition = helperXYMap(thisBoundingBox, x, y);
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.

    % Calculate scaling factors for the grid
    xscale = (max(X) - min(X)) / length(X); % Scale per x-unit
    yscale = (max(Y) - min(Y)) / length(Y); % Scale per y-unit

    % Initialize the output matrix
    drawPosition = thisBoundingBox;

    % Map bounding box coordinates to the scaled (x, y) grid
    for i = 1:size(thisBoundingBox, 1)
        % Map x and y positions
        drawPosition(i, 1) = X(int32(thisBoundingBox(i, 1))); % Map x-coordinate
        drawPosition(i, 2) = Y(int32(thisBoundingBox(i, 2))); % Map y-coordinate
        % Scale width and height
        drawPosition(i, 3) = xscale * thisBoundingBox(i, 3);  % Scaled width
        drawPosition(i, 4) = yscale * thisBoundingBox(i, 4);  % Scaled height
    end
end
