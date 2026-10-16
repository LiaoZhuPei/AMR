% pi4dqpskmod_NN.m
%
% Inputs:
%   Mod: pi/4-DQPSK Modulation structure
%   data: 1x(2*L) binary (0 and 1) input vector.
%
% Output:
%   y: Complex baseband pi/4-DQPSK signal.

function y = pi4dqpskmod_NN(Mod, data)

%% 1. Get Modulation Parameters
if Mod.M ~= 4
    error('pi/4-DQPSK modulation order (M) must be 4.');
end

sps = Mod.sps;
phaserot = Mod.phaserot; % 可以用来旋转整个星座图族
h = Mod.PulseShaping.h;
ini_phase = Mod.ini_phase;
dfcTs = Mod.dfcTs;
delay = Mod.PulseShaping.delay;
phase_map = Mod.PhaseMap; % 获取相位变化映射表

%% 2. Differential Encoding (核心逻辑)
% 将比特流每2个一组进行分组
bits_per_symbol = 2;
data_matrix = vec2mat(data, bits_per_symbol);

% 将二进制转换为十进制索引 (0, 1, 2, 3)
decimal_indices = bi2de(data_matrix, 'left-msb');

% 根据索引从PhaseMap中查找每个符号对应的相位变化
phase_changes = phase_map(decimal_indices + 1);

% --- 进行差分编码 ---
% 初始化第一个符号的相位。可以为0，或一个随机初始值。
% 这里我们用 phaserot 作为初始相位，以便外部控制。
num_symbols = length(phase_changes);
absolute_phases = zeros(1, num_symbols);
last_phase = phaserot; 

for k = 1:num_symbols
    % 当前符号的绝对相位 = 上一个符号的相位 + 本次相位变化
    current_phase = last_phase + phase_changes(k);
    
    % 将相位保持在 [-pi, pi] 范围内 (可选，但推荐)
    % absolute_phases(k) = mod(current_phase + pi, 2*pi) - pi; 
    absolute_phases(k) = current_phase; % 不做wrap-around也可以，exp(1i*...)会自动处理
    
    % 更新上一个相位
    last_phase = absolute_phases(k);
end

%% 3. Modulation
% 将计算出的绝对相位转换为复数符号
S = exp(1i * absolute_phases);

%% 4. Pulse Shaping
x = conv(upsample(S, sps), h); % 上采样和脉冲成形
x = x(delay + 1 : end - delay); % 去掉滤波器延迟

%% 5. Apply frequency and phase offset
% 注意：phaserot已经在差分编码的初始相位中使用，
% ini_phase 是在此之上的一个额外全局相位偏移。
y = complex(x .* exp(1i*2*pi*dfcTs/sps*(0:numel(x)-1) + 1i*ini_phase));

end

