% generate_qpsk_debug.m
% A self-contained script to generate, debug, and save a pi/4-QPSK signal.

clc;
clear;
close all;

% --- 1. Simulation Parameters ---
RndPhase = true;
ModType = 'pi4qpsk';
sps_i = 4;
rolloff = 0.35;
symbol_order = 'gray';
L = 1024;
SNR = 36;

% --- 2. Initialization ---
% IMPORTANT: This script assumes Initialize_Modulation_NN is on your MATLAB path
if ~exist('Initialize_Modulation_NN', 'file')
    error('The function "Initialize_Modulation_NN.m" is not found on the MATLAB path.');
end
[Mod_QPSK] = Initialize_Modulation_NN(ModType, 'sps',sps_i, 'beta',rolloff, 'symbol_order',symbol_order);

if RndPhase
    Mod_QPSK.ini_phase = 2*pi*rand(1,1);
end

dts = 0:(sps_i-1);
dt = dts(randi([1 length(dts)],[1 1]));
delay = 0;
Nsyms = ceil((delay+L+dt+1)/sps_i);
data = randi([0 1],[1 Nsyms*log2(Mod_QPSK.M)]);

% --- 3. Modulation ---
% We call the modulator function defined at the end of this file
x = pi4qpskmod_NN(Mod_QPSK, data);

% --- 4. Channel and Signal Extraction ---
y_noisy = awgn(x, SNR, 'measured');
y = y_noisy(delay+dt+(1:L));

% --- 5. Save Data ---
output_filename = 'My_Pi4QPSK_Sample_36dB.mat';
iq_data = [real(y); imag(y)];
save(output_filename, 'iq_data');
fprintf('\nSignal successfully generated and saved to: %s\n', output_filename);
fprintf('Variable name in the file: "iq_data"\n\n');

disp('Debug visualizations for pi/4-QPSK are being generated...');

% --- 6. Visualization (Optional) ---
figure('Name', 'pi/4-QPSK Analysis');
subplot(2,1,1);
scatter(real(y), imag(y), 10, 'filled');
axis square; grid on; title('Constellation');
subplot(2,1,2);
pwelch(y, [], [], [], sps_i, 'centered', 'psd');
title('Power Spectral Density');

%{
%% ========================================================================
%  Embedded Function: pi4qpskmod_NN
% =========================================================================
function y = pi4qpskmod_NN(Mod, data)
    % (This is the pi4qpskmod_NN function you provided earlier)
    sps = Mod.sps;
    h = Mod.PulseShaping.h;
    ini_phase = Mod.ini_phase;
    dfcTs = Mod.dfcTs;
    delay = Mod.PulseShaping.delay;
    constellation1_phases = Mod.Constellation1;
    constellation2_phases = Mod.Constellation2;

    bits_per_symbol = 2;
    data_matrix = vec2mat(data, bits_per_symbol);

    % The line we need to find is right here!
    decimal_indices = bi2de(data_matrix, 'left-msb');

    % #########################################################################
    % --- FINAL DEBUG CODE: Check Symbol Index Distribution ---
    % #########################################################################
    figure('Name', 'pi/4-QPSK Symbol Index Distribution');
    histogram(decimal_indices, 'BinMethod', 'integer');
    title(['Distribution of Symbol Indices for ', Mod.Type]);
    xlabel('Symbol Index (0, 1, 2, 3)');
    ylabel('Counts');
    xticks([0 1 2 3]);
    grid on;
    fprintf('--- Symbol Index Distribution for %s ---\n', Mod.Type);
    counts = histcounts(decimal_indices, -0.5:1:3.5);
    fprintf('Index 0 (e.g., dibit 00): %d times\n', counts(1));
    fprintf('Index 1 (e.g., dibit 01): %d times\n', counts(2));
    fprintf('Index 2 (e.g., dibit 10): %d times\n', counts(3)); % Gray code for '11'
    fprintf('Index 3 (e.g., dibit 11): %d times\n', counts(4)); % Gray code for '10'
    fprintf('-------------------------------------------\n');
    % #########################################################################

    num_symbols = size(data_matrix, 1);
    absolute_phases = zeros(1, num_symbols);

    for k = 1:num_symbols
        current_index = decimal_indices(k) + 1;
        if mod(k, 2) == 1
            absolute_phases(k) = constellation1_phases(current_index);
        else
            absolute_phases(k) = constellation2_phases(current_index);
        end
    end

    S = exp(1i * absolute_phases);
    x = conv(upsample(S, sps), h);
    x = x(delay + 1 : end - delay);
    y = complex(x .* exp(1i*2*pi*dfcTs/sps*(0:numel(x)-1) + 1i*ini_phase));
end
%}