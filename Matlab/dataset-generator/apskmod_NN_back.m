% --- apskmod_NN.m ---
function y = apskmod_NN(Mod, data)
%APSKMOD_NN M-APSK调制器
%   Y = APSKMOD_NN(MOD, DATA) 使用MOD结构体中定义的参数对
%   二进制数据DATA进行M-APSK调制。

%% 获取调制参数
symbolmap = Mod.Txsymbolmap;
sps = Mod.sps;
M = Mod.M;
phaserot = Mod.phaserot;
h = Mod.PulseShaping.h;
ini_phase = Mod.ini_phase;
dfcTs = Mod.dfcTs;
delay = Mod.PulseShaping.delay;
half_symbol = Mod.PulseShaping.half_symbol;

%% 信号星座图
% 调用APSK星座生成函数
Constellation = apsk_constellation_NN(M) .* exp(1i*phaserot);

%% 符号映射
% 将输入的二进制数据流(data)转换为十进制符号索引(v)
% 为了通用性，这里使用简单的二进制映射。
v = symbolmap(bi2de(vec2mat(data,log2(M)),'left-msb')+1);

%% 调制
% 根据符号索引从星座图中查找对应的复数符号
S = Constellation(v+1); % 调制器输出的符号序列

% 上采样和脉冲成形
x = conv(upsample(S,sps),h); 

% 移除由滤波器引入的多余样本，以对齐信号
x = x(delay+1-half_symbol:end-delay-half_symbol);

%% 应用频率和相位偏移
% 将载波频偏和初始相位添加到基带信号中
y = complex(x.*exp(1i*2*pi*dfcTs/sps*(0:numel(x)-1)+1i*ini_phase)); 

end