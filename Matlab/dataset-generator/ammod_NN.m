% ammod_NN.m

% 功能：对输入的消息信号进行标准调幅 (AM)
%
% 输入:
%   Mod:  由 Initialize_Modulation_NN 生成的调制结构体
%   data: 模拟基带消息信号 (行向量)
%
% 输出:
%   y:    复数基带AM信号 (行向量)

function y = ammod_NN(Mod, data)

%% 1. 从结构体中获取调制参数
sps = Mod.sps;                  % 每个输入采样点对应的输出采样点数 (上采样因子)
ini_phase = Mod.ini_phase;      % 载波初始相位
dfc_div_fs = Mod.dfc_div_fs;    % 载波频率与输出采样率之比 (fc/fs_out)
m = Mod.m;                      % 调制指数

%% 2. 准备基带信号
% 为了确保调制指数'm'的物理意义正确，我们将输入信号归一化，使其峰值为1
% 这样, m * data_norm 的峰值就是 'm'
if max(abs(data)) == 0
    data_norm = data; % 如果输入是全零，避免除以零
else
    data_norm = data / max(abs(data));
end

% 对基带信号进行上采样，以匹配输出采样率
% 使用 resample 函数，它包含一个内置的抗混叠滤波器，效果比 interp 好
data_upsampled = resample(data_norm, sps, 1);

%% 3. 生成AM信号
% 获取输出信号的长度和时间索引向量'n'
L_out = length(data_upsampled);
n = 0:(L_out - 1);

% AM信号的包络 (Envelope)
% A(t) = Ac * (1 + m * m_norm(t))
% 在仿真中，我们通常假设载波幅值 Ac = 1
envelope = 1 + m * data_upsampled;

% 检查是否过调制并给出警告
if m > 1
    warning('调制指数 m > 1，发生过调制！');
end

% 生成复数载波
% carrier = exp(j * (2*pi*fc*t + phi)) = exp(j * (2*pi*(fc/fs)*n + phi))
carrier = exp(1i * (2 * pi * dfc_div_fs * n + ini_phase));

% 将包络与载波相乘得到最终的复数基带AM信号
% 对于标准AM，信息只在I分量上，Q分量理论上为0。
% y = (Envelope + j*0) * (cos(...) + j*sin(...))
%   = Envelope * cos(...) + j * Envelope * sin(...)
% 注意：在许多SDR和仿真中，AM被表示为纯I信号，即 y = Envelope。
% 但为了与QAM代码中的频偏/相偏模型保持一致，我们生成一个带载波的复数信号。
% 如果 dfc_div_fs = 0, 这将退化为一个纯I信号。
y = envelope .* carrier;

end
