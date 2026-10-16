function tbl = countEachPixelLabel(lds)
    labels = lds.read();
    lds.reset();
    labels = labels{1};
    
    % Initialize counter
    cats = categories(labels(1));
    s = struct();
    for i = 1:numel(cats)
        s.(cats{i}) = 0;
    end
    st = s;
    
    labels = lds.readall();
    for i = 1:numel(labels)
        label = labels{i};
        for j = 1:numel(cats)
            cat = cats{j};
            numPixels = sum(label(:) == cat);
            s.(cat) = s.(cat) + numPixels;
            if (numPixels) > 0
                st.(cat) = st.(cat) + numel(label(:));
            end
        end
    end
    
    tbl = table(cats, zeros(numel(cats),1), zeros(numel(cats),1),...
        VariableNames=["Name","PixelCount","ImagePixelCount"]);
    for i = 1:numel(cats)
        tbl.PixelCount(i) = s.(cats{i});
        tbl.ImagePixelCount(i) = st.(cats{i});
    end
end
