function helperPlotROIs(Smag,f,t,inputROITable,Nvargs)
% HELPERPLOTROIS - Plot spectrogram with regions of interest (ROIs) overlaid.
%   HELPERPLOTROIS(SMAG, F, T, INPUTROITABLE, NVARGS) visualizes the time-frequency
%   magnitude spectrogram (SMAG) with overlaid bounding boxes for regions of interest
%   defined in INPUTROITABLE. Optional arguments control the display of classes and simplification.
%
%   Inputs:
%       SMAG         - Spectrogram magnitude (frequency x time matrix).
%       F            - Frequency vector (Hz).
%       T            - Time vector (s).
%       INPUTROITABLE - Table containing ROI properties, including bounding boxes.
%       NVARGS       - Name-Value pairs to customize the visualization:
%                      Classes    - (Optional) List of ROI classes to visualize.
%                      Simplify   - (Optional) Simplify display for multiple plots (default: 0).
%
% This function is only intended to support examples in the Signal
% Processing and Wavelet Toolboxes. It may change or be removed in a future
% release.

arguments
    Smag
    f
    t
    inputROITable
    Nvargs.Classes = []
    Nvargs.Simplify = 0 %Simplify for multiple display to gather
end
S = 20*log10(Smag);

%clims = [prctile(S(:),99)-50,prctile(S(:),99)+10];
max_val = max(S(:));
clims = [-10, max_val]; % 设置STFT图幅值的最小值

imagesc(t,f,S,clims)
if ~Nvargs.Simplify
    colorbar
end
title("Smag with ROIs")
ylabel("Frequency (Hz)")
xlabel("Time (s)")
hold on;
colorbar_handle = colorbar; % Add the color bar

% Add a label to the color bar
ylabel(colorbar_handle, 'Magnitude (dB)');


% Loop through each blob and draw the bounding box if it meets the size
% criteria
x = t;
y = f;

if ~isempty(Nvargs.Classes)
    classList = unique(Nvargs.Classes);
    if isnumeric(classList)
        color = [prism(5);lines(5)];
        color(5,:) = []; %Remove blue color
        classList = mat2cell(string(classList),ones(size(classList,1),1));
        classNames = classList;
    else
        color = prism(20);
        color(5,:) = []; %Remove blue color
        classNames = {"BLE_1MHz";"BLE_2MHz";"BT_classic";"WLANWBG";"WLANWN";"WLANWAC"};
    end
    
end

if ~isempty(Nvargs.Classes)
    for i = 1:length(classList)
        classcurr = classList{i};
        coloridx = find(strcmp([classNames{:}],classcurr));
        plot(NaN, NaN, 's', Color=color(coloridx,:), MarkerFaceColor=color(coloridx,:)); % Invisible markers for legend
    end
    if ~Nvargs.Simplify

        legend(classList,Location="eastoutside",Interpreter="none");  % Adjust location as needed
    end
end
% Add Bounding Box in TF Unit
if ~ismember('BoundingBoxTF',inputROITable.Properties.VariableNames)
    inputROITable.BoundingBoxTF = helperXYMap(inputROITable.BoundingBox,x,y);
end
% Add Index Box in TF Unit
if ~ismember('Index',inputROITable.Properties.VariableNames)
    inputROITable.Index = (1:height(inputROITable))';
end

for k = 1 : height(inputROITable)
    drawPosition = inputROITable{k,"BoundingBoxTF"};
    index = inputROITable{k,"Index"};
    if ~isempty(Nvargs.Classes)
        if isnumeric(Nvargs.Classes)
            classcurr = Nvargs.Classes(k);
        else
            classcurr = Nvargs.Classes{k};
        end
        coloridx = find(strcmp([classNames{:}],string(classcurr)));
        if Nvargs.Simplify
            rectangle('Position',drawPosition,EdgeColor=color(coloridx,:),LineWidth=1.5);
        else
            rectangle('Position',drawPosition,EdgeColor=color(coloridx,:),LineWidth=2);
        end
    else
        rectangle('Position',drawPosition,EdgeColor="#EDB120",LineWidth=1.5);
    end
    if ~Nvargs.Simplify
        text(drawPosition(1),drawPosition(2)+6e5,string(index),Color="w",FontWeight="bold");
    end
    hold off;
end