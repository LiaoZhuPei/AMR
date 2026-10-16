%% 纯图像分割测试脚本 (修正版：RGB转灰度)
clear; clc; close all;

%% 1. 读取图片
filename = "image_spe000.png";

if ~isfile(filename)
    error('错误：找不到文件 %s。请确认该图片就在当前 MATLAB 的工作目录中。', filename);
end

% 读取图像
rawImg = imread(filename);

% 预处理1：去掉透明通道（如果 PNG 是 RGBA 格式）
if size(rawImg, 3) == 4
    img_rgb = rawImg(:,:,1:3); 
else
    img_rgb = rawImg;
end

% 预处理2：转换为灰度图 (解决 "应为二维" 的报错)
if size(img_rgb, 3) == 3
    img_input = rgb2gray(img_rgb); % 转为 256x256 的二维矩阵
    disp('已将 RGB 图像转换为灰度图像以适配模型输入。');
else
    img_input = img_rgb; % 本身就是灰度图
end

% 打印信息
fprintf('图片加载成功！\n原始尺寸: %d x %d x %d\n输入模型尺寸: %d x %d (二维)\n', ...
    size(img_rgb,1), size(img_rgb,2), size(img_rgb,3), size(img_input,1), size(img_input,2));

% 显示原图
figure(Position=[100 100 1200 600]);
subplot(1, 2, 1);
imshow(img_rgb);
title('输入的原始图片');

%% 2. 调用 SAM 模型
disp('---------------------------------------------');
disp('正在调用 SAM 模型进行分割 (helperSegmentImageSAM)...');

try
    % 传入灰度图 (二维)
    L = helperSegmentImageSAM(img_input);
    
    % 统计结果
    labels = unique(L);
    num_regions = length(labels) - 1; 
    
    fprintf('SAM 分割完成。\n');
    fprintf('检测到的区域数量: %d\n', num_regions);
    
    %% 3. 显示结果
    subplot(1, 2, 2);
    if num_regions > 0
        % 将分割结果叠加在原图(彩色)上，方便观察
        overlayImg = labeloverlay(img_rgb, L, 'Transparency', 0.4, 'Colormap', 'jet');
        imshow(overlayImg);
        title({['分割成功！检测到 ', num2str(num_regions), ' 个目标'], '彩色区域为分割 Mask'});
    else
        imshow(img_rgb);
        title('未检测到任何目标 (结果为空)');
    end
    
catch ME
    errordlg(ME.message, '模型调用出错');
    fprintf('报错信息: %s\n', ME.message);
    % 如果出错，打印堆栈跟踪以便定位是 helper 内部哪一行报错
    disp('堆栈跟踪:');
    disp(ME.stack(1));
end