function [iq_data,fs] = read_binary_iq(filename, data_type, max_samples)
% 读取二进制 IQ 数据并归一化
    if nargin < 2 || isempty(data_type), data_type = 'float32'; end
    if nargin < 3, max_samples = inf; end

    fprintf('正在读取文件: %s ...\n', filename);
    fid = fopen(filename, 'r');
    if fid == -1, error('无法打开文件'); end

    % 读取 (I/Q 交织)
    if isinf(max_samples)
        raw = fread(fid, inf, data_type);
    else
        raw = fread(fid, 2 * max_samples, data_type);
    end
    fclose(fid);

    % 确保偶数长度
    if mod(length(raw), 2) ~= 0, raw = raw(1:end-1); end

    % 解交织 I/Q
    iq_data = complex(raw(1:2:end), raw(2:2:end));

    % 去直流
    iq_data = iq_data - mean(iq_data);

    % 功率归一化 (关键步骤)
    pwr = mean(abs(iq_data).^2);
    if pwr > 0, iq_data = iq_data / sqrt(pwr); end
    
    fs = []; % 占位
end