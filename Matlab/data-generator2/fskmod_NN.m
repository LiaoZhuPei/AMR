
% Inputs:
%   Mod: Modulation structure
%   data: 1x(log2(Mod.M)*L) binary (0 and 1) input vector.
%       Note: If data length is not a multiple of log2(Mod.M), some dummy 0 bits are appended at the end of data.
%
% Output:
%   y: Complex baseband signal as a row vector with length L*Mod.sps


function y = fskmod_NN(Mod, data)

%% Get Modulation Parameters
symbolmap = Mod.Txsymbolmap;
sps = Mod.sps;
h = Mod.h;
M = Mod.M;
phase_cont = Mod.phase_cont;
ini_phase = Mod.ini_phase;
dfcTs = Mod.dfcTs;

%% Cosntant parameters
% Initialize the phase increments and the oscillator phase for modulator with
% discontinuous phase.
phaseIncr = (2*pi*h/(2*sps))*(0:sps-1)'*(-(M-1):2:(M-1));

% phIncrSym is the incremental phase over one symbol, across all M tones.
phIncrSym = phaseIncr(end,:);

% phIncrSamp is the incremental phase over one sample, across all M tones.
phIncrSamp = phaseIncr(2,:);    % recall that phaseIncr(1,:) = 0

%% Symbol Mapping
x = symbolmap(bi2de(vec2mat(data,log2(M)),'left-msb')+1);
L = length(x);

%% Modulation
OscPhase = ini_phase*ones(1, M); % Initialization
Phase = zeros(sps*L, 1); % phase = sps*# of symbols x #

% Loop over symbols
prevPhase = ini_phase;
for iSym = 1:L
    
    % Get the initial phase for the current symbol
    if strcmpi(phase_cont,'cont')
        ph1 = prevPhase;
    else
        ph1 = OscPhase(x(iSym)+1);
    end
    
    % Compute the phase of the current symbol by summing the initial phase
    % with the per-symbol phase trajectory associated with the given M-ary
    % data element.
    Phase(sps*(iSym-1)+1:sps*iSym) = ph1*ones(sps,1) + phaseIncr(:,x(iSym)+1);
    
    % Update the oscillator for a modulator with discontinuous phase.
    % Calculate the phase modulo 2*pi so that the phase doesn't grow too
    % large.
    if strcmpi(phase_cont,'discont')
        OscPhase = rem(OscPhase + phIncrSym + phIncrSamp, 2*pi);
    end
    
    % If in continuous mode, the starting phase for the next symbol is the
    % ending phase of the current symbol plus the phase increment over one
    % sample.
    prevPhase = Phase(sps*iSym) + phIncrSamp(x(iSym)+1);
end

%% Apply frequency offset
y = exp(1i*Phase'+1i*2*pi*dfcTs/sps*(0:sps*L-1));