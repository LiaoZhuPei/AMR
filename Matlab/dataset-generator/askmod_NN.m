function y = askmod_NN(Mod, data)
%ASKMOD_NN M-ASK调制器
%   Y = ASKMOD_NN(MOD, DATA) 使用MOD结构体中定义的参数对
%   二进制数据DATA进行M-ASK调制。
%
%   MOD是一个结构体，由Initialize_Modulation_NN创建，包含以下字段：
%   - M: 调制阶数
%   - sps: 每个符号的采样点数
%   - Txsymbolmap: 符号映射表
%   - phaserot: 相位旋转
%   - PulseShaping.h: 脉冲成形滤波器
%   - PulseShaping.delay: 滤波器延迟
%   - PulseShaping.half_symbol: 半个符号的采样数
%   - ini_phase: 初始相位
%   - dfcTs: 归一化频偏

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
% 调用辅助函数生成ASK星座，并应用相位旋转
% 对于标准ASK，星座点在实轴上。相位旋转会将其旋转到复平面上。
Constellation = ask_constellation_NN(M) .* exp(1i*phaserot);

%% 符号映射
% 将输入的二进制数据流(data)转换为十进制符号索引(v)
% 'left-msb'表示将每log2(M)个比特的最高位视为最高有效位
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
