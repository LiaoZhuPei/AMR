filename = 'AAA.mat'; % 替换你的文件名

% 1. 以只读方式打开文件
fid = fopen(filename, 'r');

if fid == -1
    error('无法打开文件');
end

% 2. 读取前 128 个字节（这是标准的 MAT 文件头大小）
% *char 强制将其作为字符读取，而不是数字
header = fread(fid, [1, 128], '*char');

% 3. 关闭文件
fclose(fid);

% 4. 显示结果
disp(['文件头信息: ', header]);

% 5. 自动判断逻辑
if contains(header, 'MATLAB 7.3')
    disp('结论：这是一个 v7.3 版本文件 (HDF5格式)');
elseif contains(header, 'MATLAB 5.0')
    disp('结论：这是一个 v7 或 v6 版本文件 (默认格式)');
else
    disp('结论：其他版本 (如 v4)');
end