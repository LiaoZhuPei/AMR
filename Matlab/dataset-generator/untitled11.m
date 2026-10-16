% --- APSK调制信号流生成脚本 (V2 - 修正版) ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
targetFileSizeMB = 100; % <-- 修改这里: 设置目标文件大小 (MB)
outputFilename = sprintf('APSK_32_%dMB_A.iq', targetFileSizeMB); % 自动生成文件名
outputDataType = 'int16'; % 设置输出文件的数据类型 ('int16', 'int8', 'single')

% --- APSK调制器参数 ---
ModType = 'apsk';
M_i = 32;
sps_i = 4;
phaserot = 0;
ini_phase_i = 0;
dfcTs_i = 0;
rolloff = 0.35;
symbol_order = 'bin';
SNR = 36; % 信噪比(dB)
% ======================================================

fprintf('目标文件: %s\n', outputFilename);
fprintf('目标大小: %d MB\n', targetFileSizeMB);

% --- 1. 计算所需的样本总数 (修正后的健壮方法) ---
% 使用 switch 语句明确定义每种数据类型的字节数
switch outputDataType
    case {'int16', 'uint16'}
        bytesPerSample = 2;
    case {'int8', 'uint8'}
        bytesPerSample = 1;
    case 'single'
        bytesPerSample = 4;
    case 'double'
        bytesPerSample = 8;
    otherwise
        error('错误: 不支持的数据类型 "%s"。请使用 int16, int8, single 等。', outputDataType);
end

bytesPerIQPair = bytesPerSample * 2; % 一个I/Q对的字节数
targetTotalBytes = targetFileSizeMB * 1024 * 1024;
totalIQpairsNeeded = floor(targetTotalBytes / bytesPerIQPair);

% 安全检查: 确保计算出的目标样本数是合理的
if totalIQpairsNeeded <= 0
    error('错误: 目标文件大小为0或太小，无法生成任何样本。');
end

fprintf('数据类型: %s (%d bytes/sample)\n', outputDataType, bytesPerSample);
fprintf('需要生成的IQ对总数: %d\n', totalIQpairsNeeded);

% --- 2. 准备调制器和文件句柄 ---
% 假设 Initialize_Modulation_NN 和 apskmod_NN 函数已实现
[Mod_APSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

% 打开文件准备二进制写入
fileID = fopen(outputFilename, 'w');
if fileID == -1
    error('无法创建文件: %s。请检查文件夹权限。', outputFilename);
end

% --- 3. 循环生成、转换并写入数据 ---
totalBytesWritten = 0;
chunkSize = 16384; % 每次循环生成的样本数 (可以适当增大以提高效率)
scalingFactor = double(intmax(outputDataType)); % 缩放因子, e.g., 32767 for int16

fprintf('\n开始生成数据...\n');
while totalBytesWritten < targetTotalBytes
    
    % --- a. 生成一个数据块 ---
    Mod_APSK.ini_phase = 2*pi*rand(1,1);
    Nsyms = ceil(chunkSize / sps_i);
    data = randi([0 1], [1, Nsyms * log2(Mod_APSK.M)]);
    x = apskmod_NN(Mod_APSK, data);
    y_chunk = awgn(x, SNR, 'measured');
    
    % 安全检查: 如果调制函数返回空值，则停止
    if isempty(y_chunk)
        warning('调制函数返回空数据块，提前终止。');
        break;
    end
    
    % --- b. 数据类型转换与缩放 ---
    % 确保数据块长度不会导致最终文件超出目标大小
    remainingBytes = targetTotalBytes - totalBytesWritten;
    maxSamplesInChunk = floor(remainingBytes / bytesPerIQPair);
    
    if length(y_chunk) > maxSamplesInChunk
        y_chunk = y_chunk(1:maxSamplesInChunk);
    end
    
    % 如果计算后当前块为空，则跳出循环
    if isempty(y_chunk)
        break;
    end
    
    I_data_float = real(y_chunk);
    Q_data_float = imag(y_chunk);
    
    % 归一化以防止溢出
    max_abs = max(abs(y_chunk));
    if max_abs > 1.0
        I_data_float = I_data_float / max_abs;
        Q_data_float = Q_data_float / max_abs;
    end
    
    I_data_int = cast(I_data_float * scalingFactor, outputDataType);
    Q_data_int = cast(Q_data_float * scalingFactor, outputDataType);
    
    % --- c. 交错I和Q数据并写入文件 ---
    iq_matrix = [I_data_int; Q_data_int];
    iq_interleaved = iq_matrix(:);
    
    bytesWrittenThisChunk = fwrite(fileID, iq_interleaved, outputDataType);
    bytesWrittenThisChunk = bytesWrittenThisChunk * bytesPerSample;
    
    % --- d. 更新计数器和进度条 ---
    totalBytesWritten = totalBytesWritten + bytesWrittenThisChunk;
    fprintf('进度: %.2f%% (已写入 %.2f / %d MB)\n', ...
        (totalBytesWritten / targetTotalBytes) * 100, ...
        totalBytesWritten / (1024*1024), targetFileSizeMB);
end

% --- 4. 清理工作 ---
fclose(fileID);

fprintf('\n任务完成!\n');

% --- 5. 验证文件大小 ---
fileInfo = dir(outputFilename);
if ~isempty(fileInfo)
    actualFileSizeMB = fileInfo.bytes / (1024 * 1024);
    fprintf('已成功生成文件: %s\n', outputFilename);
    fprintf('实际文件大小: %.2f MB\n', actualFileSizeMB);
else
    fprintf('错误: 未找到生成的文件。\n');
end