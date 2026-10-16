% =========================================================================
%                restore_dc_offset_only.m
%       (一个专门用于还原仅含直流偏置信号的简单接收机)
% =========================================================================
clc, clear, close all

%% --- 步骤 1: 生成您提供的、仅含直流偏置的失真信号 ---
% (这部分代码直接来自您的脚本，以确保我们处理的是完全相同的信号)
RndPhase = true; 
ModType = 'ask'; M_i = 2; sps_i = 4; phaserot = 0; ini_phase_i = 0; 
dfcTs_i = 0; rolloff = 0.35; symbol_order = 'gray'; 
[Mod_ASK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);
if RndPhase
    Mod_ASK.ini_phase = 2*pi*rand(1,1);
end
L = 1024; SNR = 18;  
dts = 0:(sps_i-1); dt = dts(randi([1 length(dts)],[1 1]));
delay = 0; Nsyms = ceil((delay+L+dt+1)/sps_i);
data = randi([0 1],[1 Nsyms*log2(Mod_ASK.M)]);
x = askmod_NN_back(Mod_ASK, data); % 假设askmod_NN_back生成理想信号
y_noisy = awgn(x,SNR);

i_offset_min = -0.5; i_offset_max = -0.3;
q_offset_min = 0.3; q_offset_max = 0.5;
i_offset_rand = i_offset_min + (i_offset_max - i_offset_min) * rand();
q_offset_rand = q_offset_min + (q_offset_max - q_offset_min) * rand();
dc_offset = i_offset_rand + 1j * q_offset_rand;
y_with_dc = y_noisy + dc_offset;
y_distorted_signal = y_with_dc(delay+dt+(1:L));

%% ======================= 逆向还原过程 (简化版) =======================

% --- 核心还原步骤：校正直流偏移 ---
% 计算信号的平均值，这个平均值就是我们估计出的直流偏置
dc_offset_estimated = mean(y_distorted_signal);
fprintf('估计出的直流偏置: I_offset = %.4f, Q_offset = %.4f\n', real(dc_offset_estimated), imag(dc_offset_estimated));

% 从原始信号中减去这个估计出的偏置
y_final_restored = y_distorted_signal - dc_offset_estimated;

% (注意：我们不需要进行增益归一化、CFO校正、相位校正等任何其他操作)


%% ======================= 可视化对比 =======================
figure('Position', [50, 50, 1600, 800]);

% 原始失真信号
subplot(2, 2, 1);
plot(real(y_distorted_signal), 'b'); hold on; plot(imag(y_distorted_signal), 'r');
title('1. 原始失真信号 (仅含直流偏置)');
grid on; legend('I', 'Q');

subplot(2, 2, 2);
scatter(real(y_distorted_signal), imag(y_distorted_signal), 10, 'b', 'filled');
title('原始信号星座图');
grid on; axis equal;

% 最终还原信号
subplot(2, 2, 3);
plot(real(y_final_restored), 'b'); hold on; plot(imag(y_final_restored), 'r');
title('2. 最终还原的近似理想信号 (已校正直流偏置)');
grid on; legend('I', 'Q'); 

subplot(2, 2, 4);
scatter(real(y_final_restored), imag(y_final_restored), 10, 'b', 'filled');
title('还原后信号星座图');
grid on; axis equal;