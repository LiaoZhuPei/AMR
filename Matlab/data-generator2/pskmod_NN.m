% Inputs:
%   Mod: Modulation structure
%   data: 1x(log2(Mod.M)*L) binary (0 and 1) input vector.
%       Note: If data length is not a multiple of log2(Mod.M), some dummy 0 bits are appended at the end of data.
%
% Output:
%   y: Complex baseband signal as a row vector with length L*Mod.sps

function y = pskmod_NN(Mod, data)

%% Get Modulation Parameters
symbolmap = Mod.Txsymbolmap;
sps = Mod.sps;
M = Mod.M;
phaserot = Mod.phaserot;
h = Mod.PulseShaping.h;
ini_phase = Mod.ini_phase;
dfcTs = Mod.dfcTs;
delay = Mod.PulseShaping.delay;
half_symbol = Mod.PulseShaping.half_symbol;

%% Signal Constellation
Phase_Constellation = (phaserot+2*pi*(0:M-1)/M); % Phase of constellation points

%% Symbol Mapping
v = symbolmap(bi2de(vec2mat(data,log2(M)),'left-msb')+1);

%% Modulation
S = exp(1i*Phase_Constellation(v+1)); % modulator output signal
x = conv(upsample(S,sps),h); % upsampling and pulse shaping
x = x(delay+1-half_symbol:end-delay-half_symbol); % remove redundant samples

%% Apply frequency and phase offset
y = complex(x.*exp(1i*2*pi*dfcTs/sps*(0:numel(x)-1)+1i*ini_phase)); 