function helperSpectrumSensingPredictedPlotter(specImage, specInfo, ...
    predictedSpecLabel, actualSpecLabel, categories)
    figure;
    hAx = subplot(311);
    image(specImage,"Parent",hAx);
    title("Example Spectrogram");
    
    % Use the specInfo output from the transformed datastore to assign tick labels on
    % the x and y axes.
    yidx = linspace(1,size(specInfo.Frequencies,1),5);
    yticks(hAx, yidx);
    yticklabels(hAx, flipud(string(specInfo.Frequencies(round(yidx))/1e6)));
    ylabel(hAx, "Frequency (MHz)");
    
    xidx = linspace(1, size(specInfo.Time,2),11);
    xticks(hAx, xidx);
    xticklabels(hAx, string(specInfo.Time(round(xidx))*1e3));
    xlabel(hAx, "Time (ms)");
    
    % Add a colorbar.
    colorbar;
    
    % Draw the actual label mask.
    hAx = subplot(312);
    labelValues = double(actualSpecLabel{1});
    im = imagesc(labelValues, [1 numel(categories)]);
    title("Actual Labels");
    
    % Use the specInfo output from the transformed datastore to assign tick labels on
    % the x and y axes.
    yidx = linspace(1,size(specInfo.Frequencies,1),5);
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
    
    im.Parent.Colormap = flipud(cmap);
    colorbar(TickLabels=cellstr(categories),Ticks=ticks,...
        TickLength=0,TickLabelInterpreter="none");
    
    % Draw the predicted label mask.
    hAx = subplot(313);
    labelValues = double(predictedSpecLabel);
    im = imagesc(labelValues, [1 numel(categories)]);
    title("Predicted Labels");
    
    % Use the specInfo output from the transformed datastore to assign tick labels on
    % the x and y axes.
    yidx = linspace(1,size(specInfo.Frequencies,1),5);
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
    
    im.Parent.Colormap = flipud(cmap);
    colorbar(TickLabels=cellstr(categories),Ticks=ticks,...
        TickLength=0,TickLabelInterpreter="none");
end