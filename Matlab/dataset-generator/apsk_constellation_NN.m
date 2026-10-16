% --- apsk_constellation_NN.m ---
function C = apsk_constellation_NN(M)
%APSK_CONSTELLATION_NN 生成标准化的APSK星座
%   C = APSK_CONSTELLATION_NN(M) 根据调制阶数M生成星座点。
%   星座几何形状基于DVB-S2标准，以模拟真实信号特性。
%
%   目前支持 M = 16 或 32。

% --- 定义标准APSK参数 (基于DVB-S2) ---
switch M
    case 16
        % 16-APSK (DVB-S2): 2个环 (内圈4点, 外圈12点)
        radii_ratios = [1, 2.55];       % 半径比例
        points_per_ring = [4, 12];
        phase_offsets_deg = [45, 15]; % 各环的相位偏移 (度) - DVB-S2/S2X推荐

    case 32
        % 32-APSK (DVB-S2): 3个环 (内圈4点, 中圈12点, 外圈16点)
        radii_ratios = [1, 1.85, 2.8];  % 半径比例
        points_per_ring = [4, 12, 16];
        phase_offsets_deg = [45, 15, 0];% 各环的相位偏移 (度)
        
    otherwise
        error('当前仅支持 M=16 或 M=32 的APSK星座。');
end

if sum(points_per_ring) ~= M
    error('每个环上的点数之和必须等于M。');
end

% --- 生成星座点 ---
C_unnormalized = [];
phase_offsets_rad = deg2rad(phase_offsets_deg);

for i = 1:length(points_per_ring)
    radius = radii_ratios(i);
    num_points = points_per_ring(i);
    phase_offset = phase_offsets_rad(i);
    
    % 计算当前环上的点的相位
    phases = (0:num_points-1) * (2*pi/num_points) + phase_offset;
    
    % 生成复数星座点并追加到列表中
    ring_points = radius * exp(1i * phases);
    C_unnormalized = [C_unnormalized, ring_points];
end

% --- 归一化星座图，使平均功率为1 ---
avg_power = mean(abs(C_unnormalized).^2);
C = C_unnormalized / sqrt(avg_power);

% 转置为列向量以匹配MATLAB内置函数习惯
C = C.';

end