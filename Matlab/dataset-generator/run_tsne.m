clc; clear; close all;

%% 1. 准备数据
% 读取实测数据
[iq_real, ~] = read_binary_iq('E:/Project/AMR-Benchmark/PSK8-03-24-25-10h54m48s14.iq', 'int16', 100000);
iq_real = iq_real - mean(iq_real);           % 强制去直流
iq_real = iq_real / sqrt(mean(abs(iq_real).^2)); % 强制功率归一
% 生成仿真数据 (用你目前觉得最好的参数)
sim_cfg.M = 8;
sim_cfg.L = length(iq_real);
sim_cfg.dc_offset = 6.5; 
sim_cfg.gain_ripple = 0.005; 
sim_cfg.sps = 4;
sim_cfg.fs = 20e6;
sim_cfg.rolloff = 0.35;
sim_cfg.snr_db = 45;      % 哪怕这里是 60，实测可能是无限大
sim_cfg.pa_smooth = 7;  % 关键是非线性参数
sim_cfg.cfo_hz = 500;     % 加点频偏增加真实感

iq_sim = generate_vsg_signal(sim_cfg);
diagnose_features(iq_real, iq_sim)

%% 2. 运行 t-SNE 验证
%tsne_verify(iq_real, iq_sim);