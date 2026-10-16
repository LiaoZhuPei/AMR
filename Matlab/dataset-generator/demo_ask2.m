% --- ASK调制的主调用脚本 ---
clc, clear, close all

RndPhase = false; % 是否使用随机初始相位

% --- ASK调制器参数 ---
% (参数定义部分保持不变)
ModType = 'ask';
M_i = 2;         
sps_i = 4;       
phaserot = 0;    
ini_phase_i = 0; 
dfcTs_i = 0;     
rolloff = 0.35;  
symbol_order = 'gray'; 

% 假设 Initialize_Modulation_NN 函数已定义
[Mod_ASK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

if RndPhase
    Mod_ASK.ini_phase = 2*pi*rand(1,1);
end

% --- 随机选择突发信号参数 ---
L = 1024;  
SNR = 18;  

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0; 
Nsyms = ceil((delay+L+dt+1)/sps_i);
data = randi([0 1],[1 Nsyms*log2(Mod_ASK.M)]);

% --- 调用ASK调制函数 ---
x = askmod_NN_back(Mod_ASK, data);

% --- 通过信道 ---
y = awgn(x,SNR);


% =========================================================================
% <<< --- 新增：定义并生成随机直流偏置 --- >>>
% =========================================================================
% 定义直流偏置的范围，以模拟采集数据的特征。
% rand() 函数生成 [0, 1] 之间的均匀分布随机数。

% I通道偏置范围：例如，从 -0.25 到 -0.05
i_offset_min = -0.5;
i_offset_max = -0.3;

% Q通道偏置范围：例如，从 0.05 到 0.25
q_offset_min = 0.3;
q_offset_max = 0.5; % 假设Q偏置略小于I

% 从定义的范围内随机生成一个偏置值
% 公式: min + (max - min) * rand()
i_offset_rand = i_offset_min + (i_offset_max - i_offset_min) * rand();
q_offset_rand = q_offset_min + (q_offset_max - q_offset_min) * rand();

% 创建一个复数直流偏置
dc_offset = i_offset_rand + 1j * q_offset_rand;

% 将随机生成的直流偏置加到信号上
fprintf('添加随机直流偏置: I_offset = %.4f, Q_offset = %.4f\n', i_offset_rand, q_offset_rand);
y = y + dc_offset;
% =========================================================================


% --- 获取突发信号 ---
y = y(delay+dt+(1:L));


%% Visualization Section (可视化部分保持不变)
% (这部分代码无需修改)
% ... (此处省略与之前完全相同的绘图代码) ...
figure; 
plot(1:L, real(y), 'b-'); hold on; plot(1:L, imag(y), 'r-'); grid on;
title({'IQ 波形图 (带有随机直流偏置)'});
xlabel('采样点索引'); ylabel('幅度'); legend('I', 'Q'); hold off;

figure;
scatter(real(y), imag(y), 36, 'b', 'filled'); grid on; axis square;
title({'星座图 (带有随机直流偏置)'});
xlabel('I'); ylabel('Q'); hold on;
% 获取当前坐标轴的范围
x_lims = xlim; % 获取x轴的当前范围 [xmin, xmax]
y_lims = ylim; % 获取y轴的当前范围 [ymin, ymax]

% 使用 plot 函数绘制水平线 (y=0)
plot(x_lims, [0 0], '--k', 'LineWidth', 0.5);

% 使用 plot 函数绘制垂直线 (x=0)
plot([0 0], y_lims, '--k', 'LineWidth', 0.5);
hold off;
lim_max = max(abs([xlim, ylim])); axis([-lim_max, lim_max, -lim_max, lim_max]);

figure;
pwelch(y, [], [], [], sps_i, 'centered', 'psd');
title({'功率谱密度 (带有随机直流偏置)'});