function [y] = generate_vsg_signal(cfg)
% GENERATE_VSG_SIGNAL 生成高保真 8PSK 信号 (含 DC Offset & Gain Ripple)
% cfg: 配置结构体

    %% 1. 基础信号生成 (内部高倍过采样)
    target_sps = cfg.sps;   
    up_factor  = 8;         
    internal_sps = target_sps * up_factor;
    filt_span = 64; 
    
    num_syms = ceil(cfg.L / target_sps) + filt_span * 2;
    data = randi([0 cfg.M-1], num_syms, 1);
    mod_sig = pskmod(data, cfg.M, pi/cfg.M); 
    
    % 高性能 RRC 滤波器 (加 Kaiser 窗压低旁瓣)
    rrcFilter = rcosdesign(cfg.rolloff, filt_span, internal_sps, 'sqrt');
    win = kaiser(length(rrcFilter), 10)'; 
    rrcFilter = rrcFilter .* win;
    rrcFilter = rrcFilter / sum(rrcFilter); 
    
    high_res_sig = upfirdn(mod_sig, rrcFilter, internal_sps);
    
    %% 2. 添加物理损伤 (在高采样率下)
    
    % --- A. 多径/锯齿 (Multipath) ---
    if isfield(cfg, 'echo_gain') && cfg.echo_gain > 0
        d = cfg.echo_delay * up_factor; 
        g = cfg.echo_gain; 
        h_channel = [1, zeros(1, d), g];
        h_channel = h_channel / norm(h_channel);
        high_res_sig = filter(h_channel, 1, high_res_sig);
    end

    % --- B. 非线性 (PA Saturation) ---
    if isfield(cfg, 'pa_smooth')
        p = cfg.pa_smooth;
        high_res_sig = high_res_sig ./ max(abs(high_res_sig)); 
        high_res_sig = high_res_sig ./ (1 + abs(high_res_sig).^(2*p)).^(1/(2*p));
    end
    
    % --- C. 频偏 (CFO) ---
    if isfield(cfg, 'cfo_hz') && cfg.cfo_hz ~= 0
        t = (0:length(high_res_sig)-1)' / (cfg.fs * up_factor); 
        high_res_sig = high_res_sig .* exp(1j * 2 * pi * cfg.cfo_hz * t);
    end

    % --- D. 相位噪声 (Phase Noise) ---
    if isfield(cfg, 'pn_std') && cfg.pn_std > 0
        phi = cumsum(randn(size(high_res_sig)) * (cfg.pn_std / sqrt(up_factor)));
        high_res_sig = high_res_sig .* exp(1j * phi);
    end

    %% 3. 添加底噪 (AWGN)
    if isfield(cfg, 'snr_db')
        high_res_sig = awgn(high_res_sig, cfg.snr_db, 'measured');
    end

    %% 4. 降采样 (Decimation)
    delay = (filt_span * internal_sps) / 2;
    start_idx = delay + 1;
    valid_sig = high_res_sig(start_idx : end);
    y = valid_sig(1 : up_factor : end);
    
    if length(y) > cfg.L
        y = y(1:cfg.L);
    elseif length(y) < cfg.L
        y = [y; zeros(cfg.L - length(y), 1)];
    end
    
    %% 5. [关键修正] 添加 DC Offset & Gain Ripple (修正统计特征)
    
    % --- E. 直流偏置 (DC Offset / LO Leakage) ---
    % 作用：直接增加 AmpVar 和 Skewness，降低 Kurtosis
    if isfield(cfg, 'dc_offset') && cfg.dc_offset ~= 0
        % 生成一个随机相位的直流分量
        dc_val = cfg.dc_offset * exp(1j * rand() * 2 * pi);
        y = y + dc_val;
    end
    
    % --- F. 增益纹波 (Gain Ripple / AM Noise) ---
    % 作用：增加 PAPR，使得幅度不再恒定
    if isfield(cfg, 'gain_ripple') && cfg.gain_ripple > 0
        % 乘性噪声 (Multiplicative Noise)
        am_noise = 1 + randn(size(y)) * cfg.gain_ripple;
        y = y .* am_noise;
    end
    
    % 最终归一化
    y = y / sqrt(mean(abs(y).^2));
end