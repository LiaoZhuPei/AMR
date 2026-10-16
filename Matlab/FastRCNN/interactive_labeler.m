% =========================================================================
%           交互式信号标注工具 (V5.3 - 终极绘图兼容版)
% =========================================================================
function interactive_labeler()
% 将所有代码封装在一个主函数中

% --- 1. 用户配置 ---
FILE_PATH = '20171208.bin';
SAMPLE_RATE = 2.048e6;
DATA_TYPE = 'int16';
ANNOTATION_FILE = 'annotations_interactive.csv';
CLASS_ID = 0;

% --- 2. 创建图形界面 ---
fig = uifigure('Name', '交互式信号标注工具', 'WindowState', 'maximized');
gl = uigridlayout(fig, [2, 1]);
gl.RowHeight = {'1x', '3x'};
ax_overview = uiaxes(gl);
title(ax_overview, '步骤1: 在概览图上单击，选择要放大的区域');
xlabel(ax_overview, '时间 (s)');
ylabel(ax_overview, '频率 (Hz)');
ax_zoom = uiaxes(gl);
title(ax_zoom, '步骤2: 在下方高清图上画框，然后在框内双击以保存');
xlabel(ax_zoom, '时间 (s)');
ylabel(ax_zoom, '频率 (Hz)');

% --- 3. 生成并显示概览图 ---
disp('正在生成概览图，这可能需要一些时间...');
full_duration = 0;
item_size = 2;

try
    f_info = dir(FILE_PATH);
    if isempty(f_info), error('找不到文件: %s', FILE_PATH); end
    total_samples = f_info.bytes / item_size;
    full_duration = total_samples / SAMPLE_RATE;

    chunk_size = round(SAMPLE_RATE * 5);
    num_chunks = ceil(total_samples / chunk_size);
    decimation_factor = 200;
    overview_data = [];
    
    fileID = fopen(FILE_PATH, 'r');
    for i = 1:num_chunks
        fprintf('正在处理概览数据块 %d / %d\n', i, num_chunks);
        data_chunk = fread(fileID, chunk_size, ['*' DATA_TYPE]);
        if isempty(data_chunk), break; end
        overview_data = [overview_data; data_chunk(1:decimation_factor:end)];
    end
    fclose(fileID);

    overview_data_normalized = single(overview_data) / 32768.0;
    
    % --- 核心修改点 1: 分步计算和绘图 ---
    [s, f, t] = spectrogram(overview_data_normalized, kaiser(256,5), 128, 256, SAMPLE_RATE/decimation_factor, 'yaxis');
    imagesc(ax_overview, t, f, 10*log10(abs(s))); % 使用 imagesc 绘图
    set(ax_overview, 'YDir', 'normal'); % 修正Y轴方向
    colorbar(ax_overview);
    
    fprintf('概览图生成完毕！总时长: %.2f 秒\n', full_duration);
    
catch ME
    uialert(fig, sprintf('创建概览图失败: %s', ME.message), '错误');
    return;
end

% --- 4. 添加交互功能 ---
persistent current_roi;
current_roi = [];
ax_overview.ButtonDownFcn = @(src, event) overview_clicked(src, event);

% --- 嵌套函数定义 ---
function overview_clicked(ax, ~)
    pt = ax.CurrentPoint;
    t_center = pt(1,1);
    
    zoom_duration = min(full_duration * 0.05, 10);
    t_start = max(0, t_center - zoom_duration/2);
    t_end = min(full_duration, t_center + zoom_duration/2);
    
    disp('------------------------------------------');
    fprintf('正在缩放至: %.2fs - %.2fs\n', t_start, t_end);
    
    try
        fileID_zoom = fopen(FILE_PATH, 'r');
        start_sample = round(t_start * SAMPLE_RATE);
        num_samples_to_read = round((t_end - t_start) * SAMPLE_RATE);
        fseek(fileID_zoom, start_sample * item_size, 'bof');
        data_chunk_int = fread(fileID_zoom, num_samples_to_read, ['*' DATA_TYPE]);
        fclose(fileID_zoom);

        if isempty(data_chunk_int)
            disp('警告: 无法从该时间点读取数据。');
            return;
        end
        
        data_chunk_real = single(data_chunk_int) / 32768.0;
        
        cla(ax_zoom);
        % --- 核心修改点 2: 分步计算和绘图 ---
        [s_zoom, f_zoom, t_zoom] = spectrogram(data_chunk_real, kaiser(1024,5), 512, 1024, SAMPLE_RATE, 'yaxis');
        imagesc(ax_zoom, t_zoom, f_zoom, 10*log10(abs(s_zoom))); % 使用 imagesc 绘图
        set(ax_zoom, 'YDir', 'normal'); % 修正Y轴方向
        colorbar(ax_zoom);

        % 更新X轴标签以匹配真实时间
        xlim(ax_zoom, [0, t_zoom(end)]);
        ticks = get(ax_zoom, 'XTick');
        set(ax_zoom, 'XTickLabel', arrayfun(@(t) sprintf('%.2f', t + t_start), ticks, 'UniformOutput', false));
        
        title(ax_zoom, '步骤2: 在下方高清图上画框，然后在框内双击以保存');
        disp('高清图已渲染。请用鼠标在图上画一个矩形框。');
        
        if ~isempty(current_roi) && isvalid(current_roi), delete(current_roi); end
        
        current_roi = imrect(ax_zoom);
        if ~isempty(current_roi)
            addNewPositionCallback(current_roi, @(pos) roi_confirmed_for_save(pos, [t_start, t_end]));
        end

    catch ME_zoom
        fprintf('错误：在处理缩放图时发生错误: %s\n', ME_zoom.message);
    end
end

function roi_confirmed_for_save(pos, time_limits)
    disp('标注已确认！正在计算并保存...');
    
    % 将 imrect 的相对时间坐标转换为绝对时间
    pos(1) = pos(1) + time_limits(1);

    zoom_xlim = time_limits;
    zoom_ylim = [0, SAMPLE_RATE/2];
    
    x_center = pos(1) + pos(3)/2;
    y_center = pos(2) + pos(4)/2;
    
    x_center_norm = (x_center - zoom_xlim(1)) / (zoom_xlim(2) - zoom_xlim(1));
    y_center_norm = (y_center - zoom_ylim(1)) / (zoom_ylim(2) - zoom_ylim(1));
    width_norm = pos(3) / (zoom_xlim(2) - zoom_xlim(1));
    height_norm = pos(4) / (zoom_ylim(2) - zoom_ylim(1));
    
    [~, filename, ~] = fileparts(FILE_PATH);
    annotation_line = sprintf('%s.bin,%d,%.6f,%.6f,%.6f,%.6f\n', ...
                              filename, CLASS_ID, x_center_norm, y_center_norm, width_norm, height_norm);
                          
    try
        fileID_save = fopen(ANNOTATION_FILE, 'a');
        fprintf(fileID_save, annotation_line);
        fclose(fileID_save);
        
        fprintf('成功! 标注已保存至 %s\n', ANNOTATION_FILE);
        disp('操作完成！您可以继续在上方概览图中单击，以标注下一个信号。');
        
        if ~isempty(current_roi) && isvalid(current_roi), delete(current_roi); current_roi = []; end
        
    catch ME_save
        fprintf('错误：保存文件时出错: %s\n', ME_save.message);
    end
end

end % --- 主函数结束 ---