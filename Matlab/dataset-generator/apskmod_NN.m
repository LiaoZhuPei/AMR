% --- apskmod_NN.m (最终修正版) ---
function y = apskmod_NN(Mod, data)
%APSKMOD_NN M-APSK调制器
%   此版本修复了已知的技术问题，以支持长数据流生成。

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
% 关键修正1: 强制确保输入data为double类型，防止bi2de函数因数据类型问题而出错
data = double(data);
v = symbolmap(bi2de(vec2mat(data,log2(M)),'left-msb')+1);

%% 调制
% 根据符号索引从星座图中查找对应的复数符号
S = Constellation(v+1); 

% 上采样和脉冲成形
x = conv(upsample(S,sps),h); 

% 移除由滤波器引入的多余样本，以对齐信号 (保留您原始的独特裁剪逻辑)
x = x(delay+1-half_symbol:end-delay-half_symbol);

%% 应用频率和相位偏移
% 关键修正2: 强制维度一致并移除多余的complex()包裹
x = x(:); % 强制x为列向量
phase_ramp = (0:length(x)-1).'; % 创建一个匹配的列向量相位斜坡
y = x .* exp(1i*2*pi*dfcTs/sps * phase_ramp + 1i*ini_phase);

end