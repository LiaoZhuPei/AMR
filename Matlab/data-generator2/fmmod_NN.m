% fmmod_NN.m

% 功能：对输入的消息信号进行频率调制 (FM)
%
% 输入:
%   Mod:  由 Initialize_Modulation_NN 生成的调制结构体
%   data: 模拟基带消息信号 (行向量)
%
% 输出:
%   y:    复数基带FM信号 (行向量)

function y = fmmod_NN(Mod, data)

%% 1. 从结构体中获取调制参数
h = Mod.h;                      % 调制指数 (β or h = Δf / fm)
sps = Mod.sps;                  % 上采样因子 (fs_out / fs_in)
ini_phase = Mod.ini_phase;      % 载波初始相位
dfc_div_fs = Mod.dfc_div_fs;    % 归一化载波频率 (fc / fs_out)

%% 2. 准备基带信号和参数
% 假设输入信号的采样率为 Fs, 则其最大频率 fm = Fs/2
% 输出信号的采样率为 fs = Fs * sps
% 根据调制指数 h = Δf / fm，我们可以计算出最大频偏 Δf
% Δf = h * fm = h * (Fs/2)
%
% 调频灵敏度 kf 满足 Δf = kf * max(|data|)
% 如果我们将 data 归一化，使其峰值为 1, 则 kf = Δf
%
% 瞬时相位 φ(t) = 2*pi * integral(kf * data_norm(τ)) dτ
% 在离散时间下，kf * Fs = (h*Fs/2) * (1/fm) = h * Fs / (Fs/2) = 2h
% 离散相位增量: Δφ(n) = 2*pi * kf * data_norm(n) * T_in = 2*pi * kf * data_norm(n) / Fs
%                = 2*pi * (h * Fs/2) * data_norm(n) / Fs = pi * h * data_norm(n)

% 归一化输入信号，使其峰值为 1
if max(abs(data)) == 0
    data_norm = data; % 避免除以零
else
    data_norm = data / max(abs(data));
end

% 计算每个输入采样点对应的相位变化量
phase_dev_per_sample = pi * h * data_norm;

% 对相位变化量进行上采样。这里使用'zoh'(零阶保持)来模拟连续积分前的采样保持过程
phase_dev_upsampled = repelem(phase_dev_per_sample, sps);

% 核心步骤：相位积分
% 通过累加和 (cumsum) 实现离散积分
% 注意，上采样后，每个采样点的时间间隔是 T_out = 1/fs = 1/(Fs*sps)
% cumsum的结果需要除以sps来调整积分尺度，因为每个输入样本现在持续sps个输出样本
instantaneous_phase = cumsum(phase_dev_upsampled) / sps;

%% 3. 生成FM信号
% 获取输出信号的长度和时间索引向量'n'
L_out = length(instantaneous_phase);
n = 0:(L_out - 1);

% 生成复数载波项
carrier_phase = 2 * pi * dfc_div_fs * n + ini_phase;

% 合成最终的FM信号
% y(t) = Ac * exp(j * (2*pi*fc*t + φ(t) + φ0))
% Ac (载波幅度) 在仿真中通常设为 1
y = exp(1i * (carrier_phase + instantaneous_phase));

end
