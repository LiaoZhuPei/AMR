% --- APSK IQ生成脚本 (V9 - 精确复现最终版) ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
targetFileSizeMB = 100; % <-- 设置目标文件大小 (MB)
outputFilename = sprintf('APSK_32_%dMB_v9_REPLICA.iq', targetFileSizeMB);
outputDataType = 'int16';

% --- 信号参数 (与您原始脚本完全一致) ---
M = 32;
sps = 4;
rolloff = 0.35;
SNR_dB = 36;
symbol_order = 'bin'; % 确保与原始设置一致
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
chunkSymbolCount = 16384; % 每次循环处理16k个符号
scalingFactor = double(intmax(outputDataType));

% --- a. 在循环外预先生成核心组件 (高效) ---
% (i) 生成正确的32-APSK星座图
radii = [1, 2.15, 3.5]; n_points = [4, 12, 16]; phase_offsets = [pi/4, pi/12, 0];
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

% (iii) 计算裁剪参数 (!!! 精确复现的关键 !!!)
delay = (length(h)-1)/2;
half_symbol = round(sps/2);

fprintf('\n开始生成数据...\n');
while totalBytesWritten < targetTotalBytes
    
    % --- b. 生成随机比特流 ---
    NbitsToGenerate = chunkSymbolCount * bitsPerSymbol;
    data = randi([0 1], [1, NbitsToGenerate]);
    
    % --- c. 符号映射 ---
    indices = bi2de(reshape(data, bitsPerSymbol, []).', 'left-msb') + 1;
    symbols = constellation(indices);
    
    % --- d. 脉冲成形 ---
    upsampled_symbols = upsample(symbols, sps);
    x_conv = conv(upsampled_symbols, h);
    
    % --- e. 裁剪 (!!! 精确复现的关键 !!!) ---
    % 严格遵循您原始 apskmod_NN.m 中的裁剪逻辑
    x = x_conv(delay + 1 - half_symbol : end - delay - half_symbol);
    
    % --- f. 添加噪声和随机相位 ---
    y_chunk_full = awgn(x, SNR_dB, 'measured');
    y_chunk_full = y_chunk_full * exp(1j * 2 * pi * rand());
    
    % --- g. 确保不超过文件总大小 ---
    remainingBytes = targetTotalBytes - totalBytesWritten;
    maxIQpairsInChunk = floor(remainingBytes / bytesPerIQPair);
    if length(y_chunk_full) > maxIQpairsInChunk
        y_chunk_full = y_chunk_full(1:maxIQpairsInChunk);
    end
    
    if isempty(y_chunk_full), break; end
    y_chunk_clipped = y_chunk_full;

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
        (totalBytesWritten / targetTotalBytes) * 100 ...
        , totalBytesWritten / (1024*1024), targetFileSizeMB);
end

% --- 4. 清理工作 ---
fclose(fileID);
fprintf('\n任务完成!\n');