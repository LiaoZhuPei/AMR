% pi4qpskmod_NN.m
%
% Inputs:
%   Mod: pi/4-QPSK Modulation structure
%   data: 1x(2*L) binary (0 and 1) input vector.
%
% Output:
%   y: Complex baseband pi/4-QPSK signal.

function y = pi4qpskmod_NN(Mod, data)

%% 1. Get Modulation Parameters
if Mod.M ~= 4
    error('pi/4-QPSK modulation order (M) must be 4.');
end

sps = Mod.sps;
h = Mod.PulseShaping.h;
ini_phase = Mod.ini_phase;
dfcTs = Mod.dfcTs;
delay = Mod.PulseShaping.delay;
% 获取两个交替使用的星座图的相位映射表
constellation1_phases = Mod.Constellation1; % Standard QPSK constellation (0, pi/2, pi, 3pi/2)
constellation2_phases = Mod.Constellation2; % Rotated QPSK constellation (pi/4, 3pi/4, 5pi/4, 7pi/4)

%% 2. Absolute Symbol Mapping (核心逻辑)
% 将比特流每2个一组进行分组
bits_per_symbol = 2;
data_matrix = vec2mat(data, bits_per_symbol);

% 将二进制转换为十进制索引 (0, 1, 2, 3)，用于查找星座点
decimal_indices = bi2de(data_matrix, 'left-msb');

% --- 进行pi/4-QPSK绝对相位映射 ---
% 核心思想: 奇数符号用一个星座图，偶数符号用另一个旋转了pi/4的星座图
num_symbols = size(data_matrix, 1);
absolute_phases = zeros(1, num_symbols);

for k = 1:num_symbols
    current_index = decimal_indices(k) + 1; % MATLAB 索引从1开始
    
    % 判断当前是奇数符号还是偶数符号
    if mod(k, 2) == 1 % 奇数符号 (1, 3, 5, ...)
        % 使用第一个星座图 (standard QPSK)
        absolute_phases(k) = constellation1_phases(current_index);
    else % 偶数符号 (2, 4, 6, ...)
        % 使用第二个旋转了pi/4的星座图
        absolute_phases(k) = constellation2_phases(current_index);
    end
end

%% 3. Modulation
% 将计算出的绝对相位转换为复数符号
S = exp(1i * absolute_phases);

%% 4. Pulse Shaping
x = conv(upsample(S, sps), h); % 上采样和脉冲成形
x = x(delay + 1 : end - delay); % 去掉滤波器延迟

%% 5. Apply frequency and phase offset
% ini_phase 是一个全局相位偏移
y = complex(x .* exp(1i*2*pi*dfcTs/sps*(0:numel(x)-1) + 1i*ini_phase));

end
