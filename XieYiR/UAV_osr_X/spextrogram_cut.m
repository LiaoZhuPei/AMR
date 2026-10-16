clc;
clear;

%% 1. 路径设置
input_dir = 'E:\XieYiR\UAV_osr_X\twin_droneRFa\twin_droneRF\train';
output_dir = fullfile(input_dir, '..', 'train_h5_spectrograms'); % 输出目录

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 获取所有 .mat 文件
mat_files = dir(fullfile(input_dir, '*.mat'));
num_files = length(mat_files);
fprintf('共检索到 %d 个 .mat 文件，开始批量处理...\n', num_files);

%% 2. STFT 参数
fs = 80e6;
nfft = 2048;
window = hann(nfft);
noverlap = round(nfft * 0.05);

%% 3. 批量循环处理并保存为 .h5
tic;
for i = 1:num_files
    mat_path = fullfile(input_dir, mat_files(i).name);
    [~, base_name, ~] = fileparts(mat_files(i).name);
    h5_filename = fullfile(output_dir, [base_name, '.h5']);
    
    % 加载数据
    data = load(mat_path);
    x = complex(data.I, data.Q);
    
    % 计算 STFT 矩阵
    s = spectrogram(x, window, noverlap, nfft, fs, 'centered');
    
    % 转为单精度 single (节省空间)
    real_part = single(real(s));
    imag_part = single(imag(s));
    
    % 如果已经存在同名文件则先删除，避免重复写入报错
    if exist(h5_filename, 'file')
        delete(h5_filename);
    end
    
    % 写入实部与虚部至 HDF5 文件
    mat_size = size(s); % [2048, 2055]
    h5create(h5_filename, '/real', mat_size, 'Datatype', 'single');
    h5write(h5_filename, '/real', real_part);
    
    h5create(h5_filename, '/imag', mat_size, 'Datatype', 'single');
    h5write(h5_filename, '/imag', imag_part);
    
    % 【可选】：如果你后续在 Python 中只用 dB 谱，也可以直接把 P_dB 存进去：
    % p_db = single(20 * log10(abs(s)));
    % h5create(h5_filename, '/magnitude_dB', mat_size, 'Datatype', 'single');
    % h5write(h5_filename, '/magnitude_dB', p_db);
    
    if mod(i, 50) == 0 || i == num_files
        fprintf('进度: [%d/%d] 已处理完成: %s\n', i, num_files, [base_name, '.h5']);
    end
end
toc;
fprintf('所有文件处理完毕！保存路径为: %s\n', output_dir);