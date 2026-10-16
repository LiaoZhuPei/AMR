% gmskmod_NN.m
%
% Inputs:
%   Mod: GMSK Modulation structure from Initialize_Modulation_NN
%   data: 1xL binary (0 and 1) input vector.
%
% Output:
%   y: Complex baseband GMSK signal.

function y = gmskmod_NN(Mod, data)

%% 1. 获取调制参数
sps = Mod.sps;
h = Mod.h; % 调制指数, GMSK中为0.5
h_gauss = Mod.PulseShaping.h_gauss; % 高斯滤波器
delay = Mod.PulseShaping.delay;
ini_phase = Mod.ini_phase;
dfcTs = Mod.dfcTs;

%% 2. 符号映射
% GMSK是二进制的，将比特[0, 1]映射为电平[-1, 1]以代表频率的偏移方向
mapped_data = 2*data - 1;

%% 3. 生成频率脉冲并进行高斯滤波
% 创建一个理想的矩形频率脉冲序列 (NRZ)
% 每个符号对应sps个采样点的恒定电平
rectangular_pulses = upsample(mapped_data, sps);
% 用矩形窗将其变为阶梯状信号
rectangular_pulses = conv(rectangular_pulses, ones(1, sps), 'same');

% 用高斯滤波器对频率脉冲进行“脉冲成型”，使其变得平滑
frequency_shaped = conv(rectangular_pulses, h_gauss, 'full');
frequency_shaped = frequency_shaped(delay + 1 : end - delay); % 去掉滤波器延迟

%% 4. 连续相位调制 (核心)
% GMSK的本质是频率调制，相位是频率的积分
% 相位增量 = pi * h * 归一化的频率脉冲 / sps
phase_increment = (pi * h / sps) * frequency_shaped;

% 通过累加(积分)计算每个采样点的绝对相位
phase_trajectory = cumsum(phase_increment) + ini_phase;

%% 5. IQ调制
% 将相位轨迹转换为复数IQ信号
x = exp(1i * phase_trajectory);

%% 6. 应用载波频偏
% (ini_phase 已经在相位轨迹的起点加入)
y = complex(x .* exp(1i*2*pi*dfcTs/sps*(0:numel(x)-1)));

end