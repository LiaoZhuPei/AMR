%% Signal Analysis Script: pi/4-QPSK vs. pi/4-DQPSK
% This script loads simulated signal data and uses two classical methods
% to differentiate between pi/4-QPSK and pi/4-DQPSK.

clc;
clear;
close all;

%% --- 1. Configuration: Load Your Data ---
% !!! PLEASE MODIFY THESE FILENAMES to match your generated data !!!
% Let's assume you have one high-SNR sample for each modulation type.

% --- File for pi/4-QPSK ---
pi4qpsk_file = 'My_Pi4QPSK_Sample_36dB.mat'; % e.g., 'data_folder/Pi4QPSK_001_30dB.mat'

% --- File for pi/4-DQPSK ---
pi4dqpsk_file = 'My_Pi4DQPSK_Sample_36dB.mat'; % e.g., 'data_folder/Pi4DQPSK_001_30dB.mat'

% Check if files exist
if ~isfile(pi4qpsk_file)
    error('pi/4-QPSK file not found. Please check the path: %s', pi4qpsk_file);
end
if ~isfile(pi4dqpsk_file)
    error('pi/4-DQPSK file not found. Please check the path: %s', pi4dqpsk_file);
end

% Load the data
fprintf('Loading data...\n');
pi4qpsk_data_struct = load(pi4qpsk_file);
pi4dqpsk_data_struct = load(pi4dqpsk_file);

% The data is likely stored as 2x1024 (I;Q). We need to convert it to a complex signal.
% Assuming the variable name in your .mat file is 'iq_data' or similar.
% If your variable name is different, change 'iq_data' below.
pi4qpsk_iq = pi4qpsk_data_struct.iq_data;
pi4dqpsk_iq = pi4dqpsk_data_struct.iq_data;

% Convert to complex vector y = I + jQ
y_qpsk = pi4qpsk_iq(1,:) + 1i * pi4qpsk_iq(2,:);
y_dqpsk = pi4dqpsk_iq(1,:) + 1i * pi4dqpsk_iq(2,:);

fprintf('Data loaded and converted to complex signals.\n\n');

%% --- 2. Method 1: 4th Order Cyclic Cumulant (C40) ---
fprintf('--- Method 1: 4th Order Cyclic Cumulant (C40) ---\n');

% Function to calculate C40
calculate_c40 = @(y) mean(y.^4) - 3 * (mean(y.^2)).^2;

% Calculate for pi/4-QPSK
c40_qpsk = calculate_c40(y_qpsk);
fprintf('C40 for pi/4-QPSK:  %.4f + %.4fi  (Absolute value: %.4f)\n', ...
        real(c40_qpsk), imag(c40_qpsk), abs(c40_qpsk));

% Calculate for pi/4-DQPSK
c40_dqpsk = calculate_c40(y_dqpsk);
fprintf('C40 for pi/4-DQPSK: %.4f + %.4fi  (Absolute value: %.4f)\n', ...
        real(c40_dqpsk), imag(c40_dqpsk), abs(c40_dqpsk));

% Interpretation
fprintf('\nInterpretation:\n');
fprintf('  - Linear modulations (like pi/4-QPSK) should have a significant non-zero C40.\n');
fprintf('  - Non-linear/differential modulations (like pi/4-DQPSK) should have a C40 close to zero.\n');
fprintf('The results should clearly show this difference for high-SNR signals.\n\n');


%% --- 3. Method 2: Histogram of Phase Differences ---
fprintf('--- Method 2: Histogram of Phase Differences ---\n');

% --- Process pi/4-QPSK ---
% Calculate instantaneous phase
phase_qpsk = angle(y_qpsk);
% Calculate phase difference (and unwrap it)
diff_phase_qpsk = diff(unwrap(phase_qpsk));

% --- Process pi/4-DQPSK ---
% Calculate instantaneous phase
phase_dqpsk = angle(y_dqpsk);
% Calculate phase difference (and unwrap it)
diff_phase_dqpsk = diff(unwrap(phase_dqpsk));

phase_diff_threshold = deg2rad(22.5); 

% 过滤数据
filtered_diff_qpsk = diff_phase_qpsk(abs(diff_phase_qpsk) > phase_diff_threshold);
filtered_diff_dqpsk = diff_phase_dqpsk(abs(diff_phase_dqpsk) > phase_diff_threshold);


% --- Plotting (现在使用过滤后的数据) ---
figure('Name', 'Filtered Phase Difference Histograms', 'Position', [100, 100, 1200, 500]);

% Subplot for pi/4-QPSK
subplot(1, 2, 1);
% 使用过滤后的数据绘图
histogram(rad2deg(filtered_diff_qpsk), 100, 'Normalization', 'pdf');
title({'Filtered Histogram for \pi/4-QPSK', '(Inter-Symbol Transitions Only)'});
xlabel('Phase Difference (degrees)');
ylabel('Probability Density');
grid on;
xlim([-180, 180]);
xticks(-180:45:180);
ax = gca;
ax.FontSize = 12;

% Subplot for pi/4-DQPSK
subplot(1, 2, 2);
% 使用过滤后的数据绘图
histogram(rad2deg(filtered_diff_dqpsk), 100, 'Normalization', 'pdf');
title({'Filtered Histogram for \pi/4-DQPSK', '(Inter-Symbol Transitions Only)'});
xlabel('Phase Difference (degrees)');
ylabel('Probability Density');
grid on;
xlim([-180, 180]);
xticks(-180:45:180);
ax = gca;
ax.FontSize = 12;

%{
% --- Plotting ---
figure('Name', 'Phase Difference Histograms', 'Position', [100, 100, 1200, 500]);

% Subplot for pi/4-QPSK
subplot(1, 2, 1);
histogram(rad2deg(diff_phase_qpsk), 100, 'Normalization', 'pdf');
title('Histogram of Phase Differences for \pi/4-QPSK');
xlabel('Phase Difference (degrees)');
ylabel('Probability Density');
grid on;
xlim([-180, 180]);
xticks(-180:45:180);
ax = gca;
ax.FontSize = 12;

% Subplot for pi/4-DQPSK
subplot(1, 2, 2);
histogram(rad2deg(diff_phase_dqpsk), 100, 'Normalization', 'pdf');
title('Histogram of Phase Differences for \pi/4-DQPSK');
xlabel('Phase Difference (degrees)');
ylabel('Probability Density');
grid on;
xlim([-180, 180]);
xticks(-180:45:180);
ax = gca;
ax.FontSize = 12;

fprintf('A figure has been generated showing the histograms.\n');
fprintf('\nInterpretation:\n');
fprintf('  - The pi/4-QPSK plot should have a very strong peak at 0 degrees.\n');
fprintf('  - The pi/4-DQPSK plot should have four distinct peaks at -135,
-45, +45, and +135 degrees.\n');
%}