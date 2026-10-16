% --- APSK IQ生成脚本 (V11 - 忠实复现最终版) ---
% --- 本脚本的唯一目的：忠实地、反复地调用您成功的原始函数 ---
% --- 来生成一个指定大小的大文件。 ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
targetFileSizeMB = 100; % <-- 设置目标文件大小 (MB)
outputFilename = sprintf('APSK_32_%dMB_v11_FAITHFUL.iq', targetFileSizeMB);
outputDataType = 'int16';

% --- 信号参数 (用于传入您的函数) ---
M = 32;
sps = 4;
rolloff = 0.35;
SNR_dB = 36;
symbol_order = 'bin';
% ======================================================

fprintf('================================================================\n');
fprintf("V11 脚本启动：严格使用您提供的原始函数进行批量生成。\n");
fprintf('================================================================\n\n');

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
% 设定一个合理的单次处理的符号数量，以平衡效率和内存
chunkSymbolCount = 8192; % 每次循环生成8k个符号
scalingFactor = double(intmax(outputDataType));

fprintf('\n开始生成数据...\n');
while totalBytesWritten < targetTotalBytes
    
    % --- a. 在循环内重新初始化调制器 (最关键的一步) ---
    % 这严格遵循了您原始代码的成功模式：每次生成都是一个全新的开始
    [Mod_APSK] = Initialize_Modulation_NN('apsk', 'M', M, 'sps', sps, 'beta', rolloff, 'symbol_order', symbol_order);
    
    % 随机化初始相位，这也是您原始逻辑的一部分
    Mod_APSK.ini_phase = 2 * pi * rand(1,1);
    
    % --- b. 生成该数据块所需的随机比特流 ---
    NbitsToGenerate = chunkSymbolCount * bitsPerSymbol;
    data = randi([0 1], [1, NbitsToGenerate]);
    
    % --- c. 调用您原始的调制函数 ---
    % 我们完全信任这个函数会返回一个与您短样本特征一致的数据块
    y_chunk_raw = apskmod_NN(Mod_APSK, data);
    
    % --- d. 添加高斯白噪声 ---
    y_chunk_full = awgn(y_chunk_raw, SNR_dB, 'measured');
    
    % --- e. 确保最终文件大小精确 ---
    remainingBytes = targetTotalBytes - totalBytesWritten;
    maxIQpairsInChunk = floor(remainingBytes / bytesPerIQPair);
    if length(y_chunk_full) > maxIQpairsInChunk
        y_chunk_full = y_chunk_full(1:maxIQpairsInChunk);
    end
    
    if isempty(y_chunk_full), break; end
    y_chunk_clipped = y_chunk_full;

    % --- f. 转换数据类型并写入文件 ---
    max_abs = max(abs(y_chunk_clipped));
    if max_abs > 1.0, y_chunk_clipped = y_chunk_clipped / max_abs; end
    
    I_data = cast(real(y_chunk_clipped) * scalingFactor, outputDataType);
    Q_data = cast(imag(y_chunk_clipped) * scalingFactor, outputDataType);
    
    iq_interleaved = [I_data; Q_data];
    iq_interleaved = iq_interleaved(:);
    
    elementsWritten = fwrite(fileID, iq_interleaved, outputDataType);
    bytesWrittenThisChunk = elementsWritten * bytesPerSample;
    totalBytesWritten = totalBytesWritten + bytesWrittenThisChunk;

    % --- g. 更新进度 ---
    fprintf('进度: %.2f%% (已写入 %.2f / %d MB)\n', (totalBytesWritten / targetTotalBytes) * 100, totalBytesWritten / (1024*1024), targetFileSizeMB);
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

