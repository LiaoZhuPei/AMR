clc;
clear;

%% 1. 路径设置
input_dir = 'E:\XieYiR\UAV_osr_X\twin_droneRFa\twin_droneRF\train';
output_dir = fullfile(input_dir, '..', 'train_h5_csd'); % 输出目录建议区分命名

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

mat_files = dir(fullfile(input_dir, '*.mat'));
num_files = length(mat_files);
fprintf('共检索到 %d 个 .mat 文件，开始批量提取循环谱密度 (CSD)...\n', num_files);

%% 2. 循环谱计算参数
fs = 80e6;               % 采样率 80 MHz
Nfft = 512;              % 谱分辨率 (输出矩阵行数: 谱频率 f)
hop = round(Nfft * 0.25);% 75% 重叠滑动分析
win = hann(Nfft);        % 加窗抑制旁瓣泄漏

%% 3. 批量循环处理并保存为 .h5
tic;
for i = 1:num_files
    mat_path = fullfile(input_dir, mat_files(i).name);
    [~, base_name, ~] = fileparts(mat_files(i).name);
    h5_filename = fullfile(output_dir, [base_name, '.h5']);
    
    % 加载数据
    data = load(mat_path);
    x = complex(data.I, data.Q);
    x = x(:); % 转为列向量
    
    % ==========================================================
    % 高性能平滑循环互谱估计算法 (Smoothed Cyclic Periodogram)
    % ==========================================================
    % 1. 分帧加窗与时频变换
    L = length(x);
    num_frames = floor((L - Nfft) / hop) + 1;
    
    % 限制最大分析帧数，防止超大文件耗尽内存 (提取最稳定的前 4000 帧足够表征特征)
    num_frames = min(num_frames, 4000);
    
    % 构建分帧矩阵 [Nfft, num_frames]
    idx = (0:Nfft-1)' + (0:num_frames-1) * hop + 1;
    frames = x(idx) .* win;
    
    % 计算每一帧的 FFT (并进行频域中心化 fftshift)
    X = fftshift(fft(frames, Nfft, 1), 1);
    
    % 2. 计算双频互谱相关矩阵 (Bi-frequency Cross-Spectral Matrix)
    % R(f1, f2) = (X * X') / num_frames -> 大小为 [Nfft, Nfft] (BLAS 矩阵加速)
    R = (X * X') / num_frames;
    
    % 3. 坐标映射: 从双频空间 (f1, f2) 转换到循环谱空间 (f, alpha)
    % f = (f1 + f2)/2,  alpha = f1 - f2
    % 这里对双频矩阵进行对角化剪切变换，使行对应谱频率 f，列对应循环频率 alpha
    CSD = zeros(Nfft, Nfft, 'single');
    for a_idx = 1:Nfft
        diag_offset = a_idx - ceil(Nfft / 2);
        diag_vals = diag(R, diag_offset);
        if ~isempty(diag_vals)
            len_d = length(diag_vals);
            start_row = floor((Nfft - len_d) / 2) + 1;
            CSD(start_row : start_row + len_d - 1, a_idx) = diag_vals;
        end
    end
    
    % 转为单精度 single
    real_part = single(real(CSD));
    imag_part = single(imag(CSD));
    
    % 如果存在则先删除
    if exist(h5_filename, 'file')
        delete(h5_filename);
    end
    
    % 写入 HDF5 文件: 尺寸为 [512, 512]
    mat_size = size(CSD);
    h5create(h5_filename, '/real', mat_size, 'Datatype', 'single');
    h5write(h5_filename, '/real', real_part);
    
    h5create(h5_filename, '/imag', mat_size, 'Datatype', 'single');
    h5write(h5_filename, '/imag', imag_part);
    
    if mod(i, 50) == 0 || i == num_files
        fprintf('进度: [%d/%d] 已处理完成: %s\n', i, num_files, [base_name, '.h5']);
    end
end
toc;
fprintf('所有循环谱文件提取完毕！保存路径为: %s\n', output_dir);