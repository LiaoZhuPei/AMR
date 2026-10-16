% =========================================================================
%                demo_ask.m (随机相位)
% =========================================================================
clc, clear, close all

% ======================= 1. 参数配置 =======================

SimulateDistortion = true; 

% --- 不完美OOK幅度参数 ---
amplitude_on = 1.0;  
amplitude_off = 0.36; 

% --- 载波频率偏移 (CFO) ---
total_rotations_in_frame = 0.5;

% --- 【关键修改】：将初始相位设置为随机！ ---
% rand() * 2 * pi 会在 [0, 2*pi] 范围内生成一个均匀分布的随机弧度值。
% 这将完美模拟您每次截取片段时，IQ波形位置的随机性。
initial_phase_offset_rad = rand() * 2 * pi; % <<--- 这里是唯一的修改点

% --- 直流偏移 (DC Offset) --- 
dc_offset_i = 0.2;
dc_offset_q = -0.1;

% --- 全局增益 ---
global_gain = 3000; 

% =========================================================================

RndPhase = false; 

% --- ASK调制器参数 ---
ModType = 'ask';
M_i = 2;         
sps_i = 4;       
phaserot = 0;    
ini_phase_i = 0; 
dfcTs_i = 0;     
rolloff = 0.35;  
symbol_order = 'gray';

[Mod_ASK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

% --- 随机选择突发信号参数 ---
L = 1024;  
SNR = 25;  

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0; 
Nsyms = ceil((delay+L+dt+1)/sps_i);
data = randi([0 1],[1 Nsyms*log2(Mod_ASK.M)]);

% --- 调用ASK调制函数 ---
x = askmod_NN(Mod_ASK, data);

% ======================= 2. 高保真信道损伤模块 =======================
% 步骤 A: 映射到不完美OOK
map_on = (x == 1);
map_off = (x == -1);
x_imperfect_ook = zeros(size(x));
x_imperfect_ook(map_on) = amplitude_on;
x_imperfect_ook(map_off) = amplitude_off;
fprintf('已将信号映射到不完美OOK [%.2f, %.2f]。\n', amplitude_off, amplitude_on);

x_distorted = x_imperfect_ook; 

if SimulateDistortion
    fprintf('正在模拟信道损伤...\n');
    
    % 步骤 B: 应用CFO和随机的初始相位
    normalized_frequency = total_rotations_in_frame / L;
    n = 0:(length(x_imperfect_ook)-1);
    cfo_rotation_factor = exp(1i * (2 * pi * normalized_frequency * n + initial_phase_offset_rad));
    x_distorted = x_imperfect_ook .* cfo_rotation_factor;
    
    % 步骤 C: 应用直流偏移
    x_distorted = x_distorted + (dc_offset_i + 1i * dc_offset_q);
end

% --- 通过信道 (添加噪声) ---
signal_power = mean(abs(x_distorted).^2); 
snr_linear = 10^(SNR / 10);
noise_power = signal_power / snr_linear;
noise_std_dev = sqrt(noise_power / 2);
noise_i = noise_std_dev * randn(size(x_distorted));
noise_q = noise_std_dev * randn(size(x_distorted));
complex_noise = noise_i + 1i * noise_q;
y = x_distorted + complex_noise;

% --- 获取突发信号 ---
y = y(delay+dt+(1:L));

% --- 应用全局增益/缩放 ---
y = y * global_gain;
fprintf('已应用全局增益: %.0f。\n', global_gain);

% --- 模拟频谱反转 ---
y = conj(y);
fprintf('已应用频谱反转 (Q通道符号翻转)。\n');

%% 可视化部分
figure; 
plot(1:L, real(y), 'b-', 'LineWidth', 1); 
hold on; 
plot(1:L, imag(y), 'r-', 'LineWidth', 1); 
grid on;
title(['仿真的带CFO的ASK信号 IQ波形 (随机相位: ' num2str(initial_phase_offset_rad) ' rad)']);
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

figure;
scatter(real(y), imag(y), 10, 'b', 'filled'); 
grid on;
axis square; 
title('仿真的带CFO的ASK信号 星座图');
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');