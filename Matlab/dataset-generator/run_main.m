clc; clear; close all;

%% 1. 设置与读取
iq_filename = 'E:/Project/AMR-Benchmark/PSK8-03-24-25-10h54m48s14.iq'; % <--- 改成你的文件名
data_type   = 'int16';         % <--- 确认是 float32 还是 int16
real_fs     = 20e6;              % <--- 采样率
real_sps    = 4;                 % <--- 符号过采样率

% 读取数据 (如果没有文件，生成假数据演示)
if exist(iq_filename, 'file')
    [iq_real,~] = read_binary_iq(iq_filename, data_type, 100000);
else
    fprintf('演示模式：生成模拟实测数据...\n');
    % 生成一个假的高质量实测信号
    mock_cfg.M=8; mock_cfg.L=50000; mock_cfg.sps=4; mock_cfg.fs=20e6;
    mock_cfg.rolloff=0.35; mock_cfg.snr_db=60; mock_cfg.echo_gain=0.02; mock_cfg.echo_delay=5;
    iq_real = generate_vsg_signal(mock_cfg);
end

%% 2. 仿真参数配置 (调参区)

sim_cfg.M       = 8;
sim_cfg.L       = length(iq_real);
sim_cfg.sps     = real_sps;
sim_cfg.fs      = real_fs;
sim_cfg.rolloff = 0.35; % 常用: 0.22, 0.25, 0.35

% --- 关键调参指南 ---
% 1. 底噪下不去? -> 提高 snr_db (试 60~80)
sim_cfg.snr_db = 85; 

% 2. 锯齿/波纹模拟 (Multipath)
%   echo_gain: 0.01~0.1 (控制锯齿深度)
%   echo_delay: 2~10 (控制锯齿密度, 值越小波浪越宽)
sim_cfg.echo_gain  = 0.02; 
sim_cfg.echo_delay = 4;

% 3. 硬件特征
sim_cfg.pa_smooth = 3.0;   % 非线性 (越小越扁)
sim_cfg.pn_std    = 0.002; % 相位噪声 (越小星座越清晰)
sim_cfg.cfo_hz    = 200;   % 频偏

%% 3. 执行与对比
fprintf('生成仿真信号...\n');
iq_sim = generate_vsg_signal(sim_cfg);

fprintf('绘制对比图...\n');
compare_signals(iq_real, iq_sim, real_fs);

fprintf('\n--- 观察 PSD 图 (左上角) ---\n');
fprintf('1. 蓝线底噪不够低? -> 增大 sim_cfg.snr_db\n');
fprintf('2. 蓝线没有波纹?   -> 增大 sim_cfg.echo_gain\n');
fprintf('3. 蓝线波纹太密?   -> 减小 sim_cfg.echo_delay\n');