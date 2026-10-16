clc; clear; close all;

%% --- 第一步：配置与读取实测 .iq 文件 ---

% 1. 设置文件路径 (请修改为你实际的文件名)
iq_filename = 'E:/Project/AMR-Benchmark/PSK8-03-24-25-10h54m48s14.iq';  % <--- 修改这里！

% 2. 设置数据类型 (重要！)
% 如果是 Signal Hound / VSG60A 导出，通常是 'float32'
% 如果是 HackRF / RTL-SDR，通常是 'int16'
data_format = 'int16'; 

% 3. 设置已知的物理参数 (用于仿真对齐)
real_fs = 20e6;   % 采样率 20MHz (必须知道，否则 PSD 频率轴不对)
real_sps = 4;     % 符号采样率 (必须知道或估计)

% 读取数据
% 即使没有文件，为了演示代码运行，我会生成一个假文件
if ~exist(iq_filename, 'file')
    fprintf('警告：找不到文件 %s，正在生成一个临时测试文件...\n', iq_filename);
    mock_data = randn(20000, 1) + 1j*randn(20000, 1); % 假噪声
    % 模拟存成 interleaved float32
    interleaved = zeros(40000, 1);
    interleaved(1:2:end) = real(mock_data);
    interleaved(2:2:end) = imag(mock_data);
    fid = fopen('test_mock.iq', 'w'); fwrite(fid, interleaved, 'float32'); fclose(fid);
    iq_filename = 'test_mock.iq';
end

[iq_real_data] = read_binary_iq(iq_filename, data_format, 100000); % 最多读10万点

%% --- 第二步：配置仿真参数 (初始猜测) ---

sim_cfg.M = 8;              % 8PSK
sim_cfg.L = length(iq_real_data); 
sim_cfg.sps = real_sps;     
sim_cfg.fs = real_fs;         
sim_cfg.rolloff = 0.35;     % 常用默认值

% [调参区]：修改这里直到图形重合
sim_cfg.snr_db      = 60;      % 尝试调整：10 ~ 30
sim_cfg.cfo_hz      = 0;       % 尝试调整：-5000 ~ 5000
sim_cfg.pn_std      = 0;    % 尝试调整：0.001 ~ 0.05 (相位噪声)
sim_cfg.pa_smooth   = 10;     % 尝试调整：2.0 (严重压缩) ~ 10.0 (线性)
sim_cfg.iq_amp_imb  = 0;       % 尝试调整
sim_cfg.iq_phs_imb  = 0;       

%% --- 第三步：生成与对比 ---

fprintf('正在生成仿真信号...\n');
iq_sim_data = generate_vsg_signal(sim_cfg);

fprintf('正在绘制对比图...\n');
% 调用之前的对比函数
compare_signals(iq_real_data, iq_sim_data, real_fs);

%% --- 第四步：分析指导 ---
fprintf('\n--- 调参指南 ---\n');
fprintf('[图1 PDF] 如果红色比蓝色宽/平 -> 减小 pa_smooth (增加非线性) 或 减小 SNR\n');
fprintf('[图2 PSD] 如果红色底噪高 -> 减小 SNR\n');
fprintf('[图2 PSD] 如果红色谱峰宽 -> 检查 sps 设置是否正确\n');
fprintf('[图4 星座] 如果红色是圆环 -> 增加 cfo_hz (虽然盲对比只需看统计特性，不用完全对齐相位)\n');