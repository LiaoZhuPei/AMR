% --- APSK IQ生成脚本 (V17 - 终极健壮版) ---
% --- 本脚本通过更可靠的方式禁用所有清理命令，解决闪屏问题 ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
targetFileSizeMB = 100; % <-- 设置目标文件大小 (MB)
outputFilename = sprintf('APSK_32_%dMB_v17_STABLE.iq', targetFileSizeMB);
outputDataType = 'int16';
user_script_name = 'demo_apsk.m'; % <-- 确保这里是您原始脚本的正确文件名
temp_function_name = 'temp_runner_autogen_v17'; % 临时函数的文件名
% ======================================================

fprintf('================================================================\n');
fprintf("V17 脚本启动：终极健壮版，彻底解决闪屏问题。\n");
fprintf('================================================================\n\n');

if ~exist(user_script_name, 'file')
    error('错误: 找不到您的原始脚本 "%s"。请确保它与本脚本在同一个文件夹下。', user_script_name);
end

% --- 步骤 1: 创建一次性的、绝对安全的临时函数 ---
fprintf('步骤 1: 正在创建一次性的、安全的临时函数...\n');
try
    original_code = fileread(user_script_name);
    
    % --- 关键修正：使用更健壮的正则表达式，分别处理每一条清理命令 ---
    % 无论它们是否在同一行，都能被准确注释掉
    modified_code = regexprep(original_code, '^\s*clc', '% $0', 'lineanchors');
    modified_code = regexprep(modified_code, '^\s*clear', '% $0', 'lineanchors');
    modified_code = regexprep(modified_code, '^\s*close\s+all', '% $0', 'lineanchors');
    
    % 构建函数代码
    function_code = sprintf('function y = %s()\n%s\nend', temp_function_name, modified_code);
    
    % 将函数代码写入临时.m文件
    fid = fopen([temp_function_name, '.m'], 'w');
    fprintf(fid, '%s', function_code);
    fclose(fid);
    
    rehash path;
    
    fprintf('        安全的临时函数 "%s.m" 创建成功。\n\n', temp_function_name);
catch ME
    error('创建临时函数时出错: %s', ME.message);
end

% --- 步骤 2: 设置自动清理器 ---
cleanupObj = onCleanup(@() delete_temp_file(temp_function_name));
fprintf('步骤 2: 已设置自动清理器。\n\n');

fprintf('目标文件: %s\n', outputFilename);
fprintf('目标大小: %d MB\n', targetFileSizeMB);

% --- 核心参数计算 ---
bytesPerSample = 2;
bytesPerIQPair = bytesPerSample * 2;
targetTotalBytes = targetFileSizeMB * 1024 * 1024;

% --- 打开文件句柄 ---
fileID = fopen(outputFilename, 'w');
if fileID == -1, error('无法创建文件: %s。请检查文件夹权限。', outputFilename); end

% --- 循环调用临时函数并写入数据 ---
totalBytesWritten = 0;
scalingFactor = double(intmax(outputDataType));
run_count = 0;

fprintf('开始生成数据...\n');

try
    while totalBytesWritten < targetTotalBytes
        
        y = feval(temp_function_name);
        
        y_chunk_clipped = y;
        
        remainingBytes = targetTotalBytes - totalBytesWritten;
        maxIQpairsInChunk = floor(remainingBytes / bytesPerIQPair);
        if length(y_chunk_clipped) > maxIQpairsInChunk
            y_chunk_clipped = y_chunk_clipped(1:maxIQpairsInChunk);
        end
        
        if isempty(y_chunk_clipped), break; end

        max_abs = max(abs(y_chunk_clipped));
        if max_abs > 1.0, y_chunk_clipped = y_chunk_clipped / max_abs; end
        
        I_data = cast(real(y_chunk_clipped) * scalingFactor, outputDataType);
        Q_data = cast(imag(y_chunk_clipped) * scalingFactor, outputDataType);
        
        iq_interleaved = [I_data; Q_data];
        iq_interleaved = iq_interleaved(:);
        
        elementsWritten = fwrite(fileID, iq_interleaved, outputDataType);
        bytesWrittenThisChunk = elementsWritten * bytesPerSample;
        totalBytesWritten = totalBytesWritten + bytesWrittenThisChunk;
        
        run_count = run_count + 1;
        fprintf('进度: %.2f%% (已写入 %.2f / %d MB) (已调用 %d 次)\n', (totalBytesWritten / targetTotalBytes) * 100, totalBytesWritten / (1024*1024), targetFileSizeMB, run_count);
    end

catch ME
    fclose(fileID);
    fprintf('\n\n!!! SCRIPT FAILED !!!\n');
    rethrow(ME);
end

% --- 清理工作 ---
fclose(fileID);
fprintf('\n任务完成!\n');

% --- 最终验证 ---
fileInfo = dir(outputFilename);
if ~isempty(fileInfo)
    actualFileSizeMB = fileInfo.bytes / (1024 * 1024);
    fprintf('已成功生成文件: %s\n', outputFilename);
    fprintf('实际文件大小: %.2f MB\n', actualFileSizeMB);
    if abs(actualFileSizeMB - targetFileSizeMB) < 1, fprintf('文件大小符合预期。\n');
    else, fprintf('警告: 文件大小与预期有偏差！\n'); end
else, fprintf('错误: 未找到生成的文件。\n'); end


% ==================== 清理函数 ====================
function delete_temp_file(func_name)
    file_to_delete = [func_name, '.m'];
    if exist(file_to_delete, 'file')
        delete(file_to_delete);
        fprintf('\n清理工作：已成功删除临时文件 "%s"。\n', file_to_delete);
    end
    clear(func_name);
end