clear; clc; 

%% 1. 准备工作
baseDir = "E:\M_Project\FasteRCNNdata";
baseDataDir = fullfile(baseDir, "Tec");

% 请确保文件名正确
matFile = fullfile(baseDataDir, "20170515_PartG_100.mat");
if ~exist(matFile, 'file')
    error('找不到数据文件: %s', matFile);
end
load(matFile); 
Fs = 1e6;

% --- 【修正点】分两步定义参数，避免报错 ---
imSize = [256 256];
opts = labelSpectrogramOptions('windowlength', ...
    Window="hann", ...
    Overlap=75, ...
    WindowLengthMode="specify", ...
    WindowLength=512, ...
    MinimumThresholdMode="specify"); % 先不在这里写 MinimumThreshold

opts.MinimumThreshold = -10; % 单独设置这一行，确保不会报错

lblDef = signalLabelDefinition("Atom", ...
    LabelDataType="logical", ...
    LabelType="roiTimeFrequency", ...
    TimeFrequencyOptions=opts);

% 加载模型
modelFolder = "E:\M_Project\FasteRCNNdata\UAV";
modelPath = fullfile(modelFolder,"UAVDetectionNetwork","trainedUAVDetectorYoloX2025-12-04-15-34-51.mat");
if ~exist(modelPath, 'file')
    error('找不到模型文件: %s', modelPath);
end
load(modelPath); 
detector = UAVdetector;

% 检查 GPU
useGPU = canUseGPU();
if useGPU
    disp('>>> 检测到 GPU，正在使用 GPU 进行测试...');
    execEnv = 'gpu';
else
    disp('>>> 未检测到 GPU，正在使用 CPU 进行测试...');
    execEnv = 'cpu';
end

%% 2. 预热 (Warm-up)
disp('正在预热模型 (Warm-up)...');
try
    % 创建一个临时的 datastore 用于预热
    lss_warm = labeledSignalSet(signal, lblDef, SampleRate=Fs);
    [sds_warm, ~] = createDatastores(lss_warm, "Atom", ...
        TimeFrequencyMapFormat="image", TimeFrequencyImageSize=imSize, ...
        TimeFrequencyLabelFormat="xywh", TimeFrequencyIncludePartialBins=true);
    
    % 读取第一张图进行一次空跑
    if hasdata(sds_warm)
        img_warm = read(sds_warm);
        if iscell(img_warm), img_warm = img_warm{1}; end
        detect(detector, img_warm, 'ExecutionEnvironment', execEnv);
    end
catch ME
    warning('预热过程出现小问题，但不影响后续测速: %s', ME.message);
end
disp('预热完成，开始正式测速...');

%% 3. 正式测速
N = 20; % 循环20次取平均

% --- 测试 A: 预处理 (信号 -> 图片) ---
fprintf('正在测试预处理耗时...\n');
time_preprocessing = 0;
for i = 1:N
    t_start = tic;
    
    % 这里模拟完整的转换过程
    lss = labeledSignalSet(signal, lblDef, SampleRate=Fs);
    [sds, ~] = createDatastores(lss, "Atom",...
        TimeFrequencyMapFormat="image", ...
        TimeFrequencyImageSize=imSize, ...
        TimeFrequencyLabelFormat="xywh", ...
        TimeFrequencyIncludePartialBins=true);
    
    if hasdata(sds)
        image_spe = read(sds);
        if iscell(image_spe), image_spe = image_spe{1}; end
    end
    
    time_preprocessing = time_preprocessing + toc(t_start);
end
avg_pre = time_preprocessing / N;

% --- 测试 B: 推理 (图片 -> BBox) ---
fprintf('正在测试推理耗时...\n');
time_inference = 0;
for i = 1:N
    t_start = tic;
    
    [bboxes, scores, labels] = detect(detector, image_spe, 'ExecutionEnvironment', execEnv);
    
    time_inference = time_inference + toc(t_start);
end
avg_infer = time_inference / N;

%% 4. 结果报告
total_time = avg_pre + avg_infer;
max_fps = 1 / total_time;

fprintf('\n================ 性能测试报告 ================\n');
fprintf('循环次数    : %d 次\n', N);
fprintf('硬件环境    : %s\n', upper(execEnv));
fprintf('----------------------------------------------\n');
fprintf('1. 预处理 (STFT+生成图片): %.4f 秒 (%.2f ms)\n', avg_pre, avg_pre*1000);
fprintf('2. 推理 (YOLO Detect)    : %.4f 秒 (%.2f ms)\n', avg_infer, avg_infer*1000);
fprintf('----------------------------------------------\n');
fprintf('单次总耗时  : %.4f 秒 (%.2f ms)\n', total_time, total_time*1000);
fprintf('理论最大FPS : %.2f FPS\n', max_fps);
fprintf('==============================================\n');