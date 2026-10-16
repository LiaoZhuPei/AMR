function C = ask_constellation_NN(M)
%ASK_CONSTELLATION_NN 生成一个归一化功率的M-ASK星座
%   C = ask_constellation_NN(M) 为M阶ASK生成星座点。
%   M必须是2的幂。星座点位于实轴上，并进行归一化，
%   使得平均功率为1。

if mod(log2(M), 1) ~= 0
    error('调制阶数M必须是2的幂。');
end

% 生成对称的幅度级别，例如M=4时为 [-3, -1, 1, 3]
amps = -(M-1):2:(M-1);

% 计算平均功率并进行归一化
pwr = mean(amps.^2);
C = amps / sqrt(pwr);

end