% --- APSK IQ生成脚本 (V6 - 最终版) ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
targetFileSizeMB = 100; % <-- 设置目标文件大小 (MB)
outputFilename = sprintf('APSK_32_%dMB_v8.iq', targetFileSizeMB);
outputDataType = 'int16';

% --- APSK调制器参数 ---
modParams = struct(...
    'ModType', 'apsk', 'M_i', 32, 'sps_i', 4, 'phaserot', 0, ...
    'ini_phase_i', 0, 'dfcTs_i', 0, 'rolloff', 0.35, 'symbol_order', 'bin' ...
);
SNR = 36;
% ======================================================

fprintf('目标文件: %s\n', outputFilename);
fprintf('目标大小: %d MB\n', targetFileSizeMB);

% --- 1. 计算核心参数 ---
bytesPerSample = 2; % For int16
bytesPerIQPair = bytesPerSample * 2;
targetTotalBytes = targetFileSizeMB * 1024 * 1024;
bitsPerSymbol = log2(modParams.M_i);

% --- 2. 打开文件句柄 ---
fileID = fopen(outputFilename, 'w');
if fileID == -1, error('无法创建文件: %s。请检查文件夹权限。', outputFilename); end

% --- 3. 循环生成并写入数据 ---
totalBytesWritten = 0;
chunkIQpairCount = 65536; % 每次循环处理64k个IQ对
scalingFactor = double(intmax(outputDataType));

fprintf('\n开始生成数据...\n');
while totalBytesWritten < targetTotalBytes
    
    % --- a. 重新初始化调制器 (保持此最佳实践) ---
    [Mod_APSK] = Initialize_Modulation_NN(modParams.ModType, 'M', modParams.M_i, 'sps', modParams.sps_i, ...
        'phaserot', modParams.phaserot, 'ini_phase', modParams.ini_phase_i, 'dfcTs', modParams.dfcTs_i, ...
        'beta', modParams.rolloff, 'symbol_order', modParams.symbol_order);
    Mod_APSK.ini_phase = 2*pi*rand(1,1);
    
    % --- b. 生成数据 ---
    % 为了确保有足够的稳定信号，生成稍多一些的符号
    NsymsToGenerate = ceil(chunkIQpairCount / modParams.sps_i) + 200; % +200 符号作为裕量
    NbitsToGenerate = NsymsToGenerate * bitsPerSymbol;
    data = randi([0 1], [1, NbitsToGenerate]);
    
    x = apskmod_NN(Mod_APSK, data);
    y_chunk_full = awgn(x, SNR, 'measured');

    % --- c. 裁剪出稳定信号 ---
    % 从信号后半段取样，避开滤波器启动的瞬态效应
    if length(y_chunk_full) > chunkIQpairCount
        startIndex = length(y_chunk_full) - chunkIQpairCount + 1;
        y_chunk_clipped = y_chunk_full(startIndex:end);
    else
        y_chunk_clipped = y_chunk_full;
    end
    
    % --- d. 确保不超过文件总大小 ---
    remainingBytes = targetTotalBytes - totalBytesWritten;
    maxIQpairsInChunk = floor(remainingBytes / bytesPerIQPair);
    if length(y_chunk_clipped) > maxIQpairsInChunk
        y_chunk_clipped = y_chunk_clipped(1:maxIQpairsInChunk);
    end
    
    if isempty(y_chunk_clipped), break; end

    % --- e. 转换并写入 ---
    max_abs = max(abs(y_chunk_clipped));
    if max_abs > 1.0, y_chunk_clipped = y_chunk_clipped / max_abs; end
    
    I_data = cast(real(y_chunk_clipped) * scalingFactor, outputDataType);
    Q_data = cast(imag(y_chunk_clipped) * scalingFactor, outputDataType);
    
    iq_interleaved = [I_data; Q_data];
    iq_interleaved = iq_interleaved(:);
    
    elementsWritten = fwrite(fileID, iq_interleaved, outputDataType);
    bytesWrittenThisChunk = elementsWritten * bytesPerSample;
    totalBytesWritten = totalBytesWritten + bytesWrittenThisChunk;

    % --- f. 更新进度 ---
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