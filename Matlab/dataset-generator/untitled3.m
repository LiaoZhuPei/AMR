% --- APSK IQ生成脚本 (V7 - 最终自洽版) ---
% --- (不依赖任何外部自定义函数) ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
targetFileSizeMB = 100; % <-- 设置目标文件大小 (MB)
outputFilename = sprintf('APSK_32_%dMB_v7_FINAL.iq', targetFileSizeMB);
outputDataType = 'int16';

% --- 信号参数 ---
M = 32;           % 调制阶数
sps = 4;          % 每个符号的采样点数
rolloff = 0.35;     % 滚降系数
SNR_dB = 36;        % 信噪比(dB)
% ======================================================

fprintf('目标文件: %s\n', outputFilename);
fprintf('目标大小: %d MB\n', targetFileSizeMB);

% --- 1. 核心参数计算 ---
bytesPerSample = 2; % For int16
bytesPerIQPair = bytesPerSample * 2;
targetTotalBytes = targetFileSizeMB * 1024 * 1024;
bitsPerSymbol = log2(M);

% --- 2. 打开文件句柄 ---
fileID = fopen(outputFilename, 'w');
if fileID == -1, error('无法创建文件: %s。请检查文件夹权限。', outputFilename); end

% --- 3. 循环生成并写入数据 ---
totalBytesWritten = 0;
chunkIQpairCount = 65536; % 每次循环处理64k个IQ对
scalingFactor = double(intmax(outputDataType));

% --- a. 在循环外预先生成核心组件 (高效) ---
% (i) 生成正确的32-APSK星座图
radii = [1, 2.15, 3.5];
n_points = [4, 12, 16];
phase_offsets = [pi/4, pi/12, 0];
constellation = [];
for i = 1:length(radii)
    angles = (0:n_points(i)-1) * (2*pi/n_points(i)) + phase_offsets(i);
    constellation = [constellation, radii(i) * exp(1j * angles)];
end
avg_power = mean(abs(constellation).^2);
constellation = constellation / sqrt(avg_power);

% (ii) 生成脉冲成形滤波器
filter_span_in_symbols = 8;
h = rcosdesign(rolloff, filter_span_in_symbols, sps, 'sqrt');

fprintf('\n开始生成数据...\n');
while totalBytesWritten < targetTotalBytes
    
    % --- b. 生成随机比特流 ---
    % 为了确保有足够的稳定信号，生成稍多一些的符号
    NsymsToGenerate = ceil(chunkIQpairCount / sps) + filter_span_in_symbols;
    NbitsToGenerate = NsymsToGenerate * bitsPerSymbol;
    data = randi([0 1], [1, NbitsToGenerate]);
    
    % --- c. 符号映射 ---
    indices = bi2de(reshape(data, bitsPerSymbol, []).', 'left-msb') + 1;
    symbols = constellation(indices);
    
    % --- d. 脉冲成形 ---
    upsampled_symbols = upsample(symbols, sps);
    x = conv(upsampled_symbols, h);
    
    % --- e. 添加噪声和随机相位 ---
    y_chunk_full = awgn(x, SNR_dB, 'measured');
    y_chunk_full = y_chunk_full * exp(1j * 2 * pi * rand()); % 随机化初始相位
    
    % --- f. 精确裁剪出稳定信号 ---
    % 从信号中间部分取样，完全避开两端的滤波器瞬态效应
    total_delay = (length(h)-1)/2;
    stable_signal = y_chunk_full(total_delay + 1 : end - total_delay);
    
    if length(stable_signal) > chunkIQpairCount
        y_chunk_clipped = stable_signal(1:chunkIQpairCount);
    else
        y_chunk_clipped = stable_signal;
    end
    
    % --- g. 确保不超过文件总大小 ---
    remainingBytes = targetTotalBytes - totalBytesWritten;
    maxIQpairsInChunk = floor(remainingBytes / bytesPerIQPair);
    if length(y_chunk_clipped) > maxIQpairsInChunk
        y_chunk_clipped = y_chunk_clipped(1:maxIQpairsInChunk);
    end
    
    if isempty(y_chunk_clipped), break; end

    % --- h. 转换并写入 ---
    max_abs = max(abs(y_chunk_clipped));
    if max_abs > 1.0, y_chunk_clipped = y_chunk_clipped / max_abs; end
    
    I_data = cast(real(y_chunk_clipped) * scalingFactor, outputDataType);
    Q_data = cast(imag(y_chunk_clipped) * scalingFactor, outputDataType);
    
    iq_interleaved = [I_data; Q_data];
    iq_interleaved = iq_interleaved(:);
    
    elementsWritten = fwrite(fileID, iq_interleaved, outputDataType);
    bytesWrittenThisChunk = elementsWritten * bytesPerSample;
    totalBytesWritten = totalBytesWritten + bytesWrittenThisChunk;

    % --- i. 更新进度 ---
    fprintf('进度: %.2f%% (已写入 %.2f / %d MB)\n', ...
        (totalBytesWritten / targetTotalBytes) * 100, ...
        totalBytesWritten / (1024*1024), targetFileSizeMB);
end

% --- 4. 清理工作 ---
fclose(fileID);
fprintf('\n任务完成!\n');

% --- 5. 最终验证 ---
fileInfo = dir(outputFilename);
if ~isempty(fileInfo)
    actualFileSizeMB = fileInfo.bytes / (1024 * 1024);
    fprintf('已成功生成文件: %s\n', outputFilename);
    fprintf('实际文件大小: %.2f MB\n', actualFileSizeMB);
    if abs(actualFileSizeMB - targetFileSizeMB) < 1
        fprintf('文件大小符合预期。\n');
    else
        fprintf('警告: 文件大小与预期有偏差！\n');
    end
else
    fprintf('错误: 未找到生成的文件。\n');
end

% --- 6. (可选) 绘制最后一个数据块的星座图以供验证 ---
figure;
scatter(real(y_chunk_clipped), imag(y_chunk_clipped), 10, 'filled');
title('最后一个生成数据块的星座图 (验证)');
axis square; grid on;
xlabel('I'); ylabel('Q');