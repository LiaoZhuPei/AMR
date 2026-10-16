function helperObjectDetectionPlotter(specImage, specInfo, specLabel, categories)
    hFig = figure;
    hAx = axes(hFig);
    image(specImage,"Parent",hAx);
    
    % Use the specInfo output from the transformed datastore to assign tick labels
    % on the x and y axes.
    yidx = linspace(1,size(specInfo.Frequencies,1),9);
    yticks(hAx, yidx);
    yticklabels(hAx, flipud(string(specInfo.Frequencies(round(yidx))/1e6)));
    ylabel(hAx, "Frequency (MHz)");
    
    xidx = linspace(1, size(specInfo.Time,2),11);
    xticks(hAx, xidx);
    xticklabels(hAx, string(specInfo.Time(round(xidx))*1e3));
    xlabel(hAx, "Time (ms)");
    
    % Draw a rectangle on the axis for each label. Assign the associated label
    % and color by label value.
    categoryColors = ["red","k","yellow","green","blue","magenta"];
    for i = 1:numel(specLabel{2})
        hR = drawrectangle(hAx, "Position", specLabel{1}(i,:));
        hR.InteractionsAllowed = "none";
        hR.Label = string(specLabel{2}(i));
        hR.Color = categoryColors(hR.Label == categories);
        hR.FaceAlpha = 0;
    end
end

