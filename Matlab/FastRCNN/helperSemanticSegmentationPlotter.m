function helperSemanticSegmentationPlotter(specImage, specInfo, specLabel, ...
    categories)
    figure;
    hAx = subplot(211);
    image(specImage,"Parent",hAx);
    
    % Use the specInfo output from the transformed datastore to assign tick label
    % on the x and y axes.
    yidx = linspace(1,size(specInfo.Frequencies,1),9);
    yticks(hAx, yidx);
    yticklabels(hAx, flipud(string(specInfo.Frequencies(round(yidx))/1e6)));
    ylabel(hAx, "Frequency (MHz)");
    
    xidx = linspace(1, size(specInfo.Time,2),11);
    xticks(hAx, xidx);
    xticklabels(hAx, string(specInfo.Time(round(xidx))*1e3));
    xlabel(hAx, "Time (ms)");
    
    % Add a colorbar.
    colorbar;

    title("Spectrogram Image");
    
    % Draw the label output mask.
    hAx = subplot(212);
    labelValues = double(specLabel{1});
    im = imagesc(labelValues, [1 numel(categories)]);
    
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
    
    % Add a colorbar that identifies the label category by color.
    cmap = cool(numel(categories));
    ticks = 1:numel(categories);
    
    im.Parent.Colormap = cmap;
    colorbar(TickLabels=cellstr(categories),Ticks=ticks,...
        TickLength=0,TickLabelInterpreter="none");

    title("Ground Truth Labels");
end