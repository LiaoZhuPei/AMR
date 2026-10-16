% --- APSK调制的主调用脚本 (仿造ASK版本) ---
clc, clear, close all

RndPhase = true; % 是否使用随机初始相位

% --- APSK调制器参数 ---
% M: 调制阶数 (e.g., 16 (default), 32)
% phaserot: 调制的相位旋转 (弧度, 默认: 0)
% sps: 每个符号的采样点数 (默认: 8)
% ini_phase: 载波初始相位 (默认: 0)
% dfcTs: 载波频偏与符号周期的乘积 (默认: 0)
% beta: 根升余弦脉冲的滚降系数 (默认: 0.35)
% symbol_order: 符号顺序 ('bin' (default) | 'gray')
ModType = 'apsk';       % <-- 修改: 调制类型
M_i = 32;               % <-- 修改: APSK调制阶数, 16-APSK是常用标准
sps_i = 4;              % 每个符号的采样点数
phaserot = 0;           % 相位旋转
ini_phase_i = 0;        % 初始相位
dfcTs_i = 0;            % 归一化频偏
rolloff = 0.35;         % 滚降系数
symbol_order = 'bin';   % <-- 修改: 对于APSK, 二进制映射更直接且不影响IQ统计特性

% 使用与代码2中类似的初始化函数来创建调制器对象
% 假设 Initialize_Modulation_NN 函数已根据代码2的逻辑实现
[Mod_APSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

% 设置随机相位
if RndPhase
    Mod_APSK.ini_phase = 2*pi*rand(1,1);
end

% --- 随机选择突发信号参数 ---
L = 13107200;  % 突发长度
SNR = 36;  % 信噪比 (dB)，一个较高的值以清晰地观察星座图

% 随机采样时间偏移
dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));

delay = 0; % 延迟 (此处为0)

% 计算所需的符号数
Nsyms = ceil((delay+L+dt+1)/sps_i);

% 生成随机的二进制信息比特流
% log2(Mod_APSK.M) 会自动根据 M_i 计算每个符号的比特数 (例如, 16-APSK为4比特)
data = randi([0 1],[1 Nsyms*log2(Mod_APSK.M)]);

% --- 调用APSK调制函数 ---
x = apskmod_NN(Mod_APSK, data); % <-- 修改: 调用APSK调制函数

% --- 通过信道 ---
y = awgn(x, SNR, 'measured'); % 使用'measured'确保信噪比精确

% --- 获取突发信号 ---
y = y(delay+dt+(1:L));

%% ================= Visualization Section =================
% 此部分代码是通用的，无需修改逻辑，只需更新标题即可

% 1. 绘制IQ波形图
figure; % 创建一个新的图形窗口
plot(1:L, real(y), 'b-', 'LineWidth', 1); % 绘制I分量
hold on; % 保持当前图形，以便在上面叠加新的曲线
plot(1:L, imag(y), 'r-', 'LineWidth', 1); % 绘制Q分量
grid on;
title(sprintf('APSK-%d IQ 波形图 (SNR = %d dB)', M_i, SNR)); % <-- 修改: 更新标题
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

% 2. 绘制星座图
figure;
scatter(real(y), imag(y), 36, 'b', 'filled'); % 'b'代表蓝色, 'filled'填充点
grid on;
axis square; % 使坐标轴比例相同，星座图更直观
title(sprintf('APSK-%d 星座图 (SNR = %d dB)', M_i, SNR)); % <-- 修改: 更新标题
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
% 设定坐标轴范围，使其围绕原点对称，看得更清楚
lim_max = max(abs([xlim, ylim])) * 1.1; % 稍微留出一些边距
axis([-lim_max, lim_max, -lim_max, lim_max]);

% 3. 绘制功率谱密度 (PSD)
% 采样率 fs 等于每个符号的采样点数 sps_i (当符号速率归一化为1时)
fs = sps_i; 

figure;
% 使用 pwelch 函数来估计PSD，它比简单的FFT更平滑、准确
pwelch(y, [], [], [], fs, 'centered', 'psd');
title(sprintf('APSK-%d 功率谱密度', M_i)); % <-- 修改: 更新标题
% pwelch函数会自动处理坐标轴标签