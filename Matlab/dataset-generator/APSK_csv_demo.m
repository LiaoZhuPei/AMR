% --- APSK调制与批量样本生成脚本 ---
clc;
clear;
close all;

% ==================== 用户配置区 ====================
N = 50; % <-- 修改这里: 设置您想一次性生成的样本总数
outputDir = 'apsk_samples'; % 设置保存样本的文件夹名称
% ======================================================

% --- 检查并创建输出文件夹 ---
if ~exist(outputDir, 'dir')
   mkdir(outputDir);
   fprintf('文件夹 "%s" 已创建。\n', outputDir);
end

fprintf('开始生成 %d 个 APSK 样本...\n', N);

% --- 主生成循环 ---
for i = 1:N
    % --- APSK调制器参数 ---
    % M: 调制阶数 (e.g., 16 (default), 32)
    % phaserot: 调制的相位旋转 (弧度, 默认: 0)
    % sps: 每个符号的采样点数 (默认: 8)
    % ini_phase: 载波初始相位 (默认: 0)
    % dfcTs: 载波频偏与符号周期的乘积 (默认: 0)
    % beta: 根升余弦脉冲的滚降系数 (默认: 0.35)
    % symbol_order: 符号顺序 ('bin' (default) | 'gray')
    ModType = 'apsk';       % 调制类型
    M_i = 32;               % APSK调制阶数
    sps_i = 4;              % 每个符号的采样点数
    phaserot = 0;           % 相位旋转
    ini_phase_i = 0;        % 初始相位
    dfcTs_i = 0;            % 归一化频偏
    rolloff = 0.35;         % 滚降系数
    symbol_order = 'bin';   % 二进制映射

    % 假设 Initialize_Modulation_NN 函数已实现
    [Mod_APSK] = Initialize_Modulation_NN(ModType,'M',M_i,'sps',sps_i,'phaserot',phaserot,'ini_phase',ini_phase_i,'dfcTs',dfcTs_i,'beta',rolloff,'symbol_order',symbol_order);

    % 设置随机相位
    RndPhase = true; % 是否使用随机初始相位
    if RndPhase
        Mod_APSK.ini_phase = 2*pi*rand(1,1);
    end

    % --- 随机选择突发信号参数 ---
    L = 1024;  % 突发长度 (最终CSV文件的列数)
    SNR = 36;  % 信噪比 (dB)

    % 随机采样时间偏移
    dts = 0:(sps_i-1);
    dt = dts(randi([1 length(dts)],[1 1]));

    delay = 0; % 延迟

    % 计算所需的符号数
    Nsyms = ceil((delay+L+dt+1)/sps_i);

    % 生成随机的二进制信息比特流
    data = randi([0 1],[1 Nsyms*log2(Mod_APSK.M)]);

    % --- 调用APSK调制函数 ---
    % 假设 apskmod_NN 函数已实现
    x = apskmod_NN(Mod_APSK, data);

    % --- 通过信道 ---
    y = awgn(x, SNR, 'measured');

    % --- 获取突发信号 ---
    y = y(delay+dt+(1:L));

    % --- 数据格式化与保存 ---
    % 1. 将I和Q分量分离
    I_data = real(y);
    Q_data = imag(y);

    % 2. 垂直堆叠成 (2, 1024) 的矩阵
    output_data = [I_data; Q_data];

    % 3. 创建文件名并保存为 .csv 文件
    filename = fullfile(outputDir, sprintf('apsk_%d_sample_%04d.csv', M_i, i));
    writematrix(output_data, filename);
    
    % 4. 在命令窗口显示进度
    if mod(i, 10) == 0 || i == N
        fprintf('已生成并保存样本: %d / %d\n', i, N);
    end
end

fprintf('\n任务完成！ %d 个样本已成功保存在文件夹 "%s" 中。\n', N, outputDir);

%% ================= Visualization Section (显示最后一个样本) =================
% 此部分代码将显示循环中最后一个生成样本的图形以供验证

% 1. 绘制IQ波形图
figure;
plot(1:L, real(y), 'b-', 'LineWidth', 1);
hold on;
plot(1:L, imag(y), 'r-', 'LineWidth', 1);
grid on;
title(sprintf('APSK-%d IQ 波形图 (最后一个样本, SNR = %d dB)', M_i, SNR));
xlabel('采样点索引 (Sample Index)');
ylabel('幅度 (Amplitude)');
legend('I 分量 (In-phase)', 'Q 分量 (Quadrature)');
hold off;

% 2. 绘制星座图
figure;
scatter(real(y), imag(y), 36, 'b', 'filled');
grid on;
axis square;
title(sprintf('APSK-%d 星座图 (最后一个样本, SNR = %d dB)', M_i, SNR));
xlabel('I 分量 (In-phase)');
ylabel('Q 分量 (Quadrature)');
lim_max = max(abs([xlim, ylim])) * 1.1;
axis([-lim_max, lim_max, -lim_max, lim_max]);

% 3. 绘制功率谱密度 (PSD)
fs = sps_i;
figure;
pwelch(y, [], [], [], fs, 'centered', 'psd');
title(sprintf('APSK-%d 功率谱密度 (最后一个样本)', M_i));