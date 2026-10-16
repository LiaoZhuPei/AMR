% =========================================================================
%               文件结构探查脚本 (probe_file.m)
% =========================================================================
clear; clc; close all;

% --- 配置 ---
FILE_PATH = '20171208.bin'; % 您的文件名

% --- 探查开始 ---
disp(['--- 正在探查文件: ', FILE_PATH, ' ---']);

try
    fileID = fopen(FILE_PATH, 'r');
    if fileID == -1
        error('无法打开文件。请检查文件名和路径是否正确。');
    end
    
    % --- 探查1: 尝试读取前16个字节作为 8位有符号整数 (int8) ---
    frewind(fileID); % 确保从文件头开始
    data_int8 = fread(fileID, 16, '*int8');
    disp('前16个字节 (作为 int8):');
    disp(data_int8'); % 转置后显示为一行
    
    % --- 探查2: 尝试读取前8个字节作为 16位有符号整数 (int16) ---
    frewind(fileID);
    data_int16 = fread(fileID, 8, '*int16');
    disp('前16个字节 (作为 int16):');
    disp(data_int16');

    % --- 探查3: 尝试读取前4个字节作为 32位单精度浮点数 (single/float32) ---
    frewind(fileID);
    data_single = fread(fileID, 4, '*single');
    disp('前16个字节 (作为 single/float32):');
    disp(data_single');
    
    fclose(fileID);
    
    disp('--- 探查完毕 ---');
    disp('请将以上打印出的三组数值回复给我。');
    
catch ME
    disp(['探查过程中发生错误: ', ME.message]);
    if exist('fileID', 'var') && fileID ~= -1
        fclose(fileID);
    end
end