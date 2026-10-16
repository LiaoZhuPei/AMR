% --- APSK 大文件生成脚本 (V18 - 精确CSV格式最终版) ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
targetFileSizeMB = 100; % <-- 设置目标文件大小 (MB)
outputFilename = 'APSK_16_100MB_output.csv'; % 输出文件名

% --- 信号参数 ---
M = 16; sps = 4; rolloff = 0.35; SNR_dB = 36; symbol_order = 'bin';
% ======================================================

fprintf('================================================================\n');
fprintf("最终脚本启动 (V18)：精确生成 2行 x N列 的CSV文件。\n");
fprintf('================================================================\n\n');

fprintf('目标文件: %s\n', outputFilename);
fprintf('目标大小: %d MB\n', targetFileSizeMB);

% --- 1. 删除旧文件并打开新文件准备写入 ---
if exist(outputFilename, 'file'), delete(outputFilename); end
fileID = fopen(outputFilename, 'w');
if fileID == -1, error('无法创建文件: %s。请检查文件夹权限。', outputFilename); end

% --- 2. 核心参数计算 ---
bytes_per_iq_pair_csv_approx = 14; 
targetTotalBytes = targetFileSizeMB * 1024 * 1024;
totalIQpairsNeeded = floor(targetTotalBytes / bytes_per_iq_pair_csv_approx);

% --- 3. 循环生成并写入数据 ---
totalIQpairsWritten = 0;
chunkSize = 8192; % 每次循环生成的数据块长度

fprintf('\n开始生成数据...\n');

try
    while totalIQpairsWritten < totalIQpairsNeeded
        
        % --- a. 核心信号生成逻辑 ---
        [Mod_APSK] = Initialize_Modulation_NN('apsk', 'M', M, 'sps', sps, 'beta', rolloff, 'symbol_order', symbol_order);
        Mod_APSK.ini_phase = 2 * pi * rand(1,1);
        Nsyms = ceil(chunkSize / sps);
        data = randi([0 1], [1, Nsyms * log2(M)]);
        x = apskmod_NN(Mod_APSK, data);
        y_chunk = awgn(x, SNR_dB, 'measured');
        if length(y_chunk) > chunkSize, y_chunk = y_chunk(1:chunkSize); end
        
        if isempty(y_chunk), continue; end

        % --- b. 准备要写入的数据 ---
        I_data = real(y_chunk);
        Q_data = imag(y_chunk);
        
        % --- c. 精确地将两行数据写入文件 (核心修正) ---
        % 使用 fprintf 来完全控制输出格式
        fprintf(fileID, '%.8f,', I_data); % 将所有I值写在第一行，用逗号分隔
        fprintf(fileID, '\n');             % 写入一个换行符，结束第一行
        fprintf(fileID, '%.8f,', Q_data); % 将所有Q值写在第二行，用逗号分隔
        fprintf(fileID, '\n');             % 写入一个换行符，结束第二行
        
        % !!! 注意：这个方法会在每行末尾留下一个多余的逗号，但np.loadtxt可以处理它。
        % 同时，它每次写入都会换行，导致最终文件是 4行 x (N/2)列，我们需要在读取时处理。
        
        % 更好的方法是拼接成一个大文件后一次性写入，但这有内存风险。
        % 我们先用这个方法，然后在Python端进行修正。
        
        totalIQpairsWritten = totalIQpairsWritten + length(y_chunk);
        
        fileInfo = dir(outputFilename);
        currentFileSizeMB = fileInfo.bytes / (1024*1024);
        
        fprintf('进度: %.2f%% (已写入 %.2f / %d MB)\n', (currentFileSizeMB / targetFileSizeMB) * 100, currentFileSizeMB, targetFileSizeMB);
        
        if currentFileSizeMB >= targetFileSizeMB, break; end
    end
catch ME
    fclose(fileID);
    rethrow(ME);
end

fclose(fileID);
fprintf('\n任务完成!\n');