% Inputs:
%   ModType: Modulation type
%       Digital Modulations: 'msk', 'gmsk', 'fsk', 'psk', 'dpsk', 'oqpsk', 'qam', 'ask'
%       Analog Modulations: 'am', 'dsb', 'usb', 'lsb', 'isb', 'fm', 'pm'
%   varargin: 2xL input arguments that can be used to assign non-default values to some fields.
%       For i=1,2,3,...,L, varargin{2*i-1} is the desired field name and
%       varargin{2*i} is the desired field value.
%
% Output:
%   Mod: Modulation structure
%   ErrorMsg: Possible error message

function Mod = Initialize_Modulation_NN(ModType,varargin)

switch lower(ModType)


    case 'fsk'

        % M: Modulation order (2 (default), 4, or 8)
        % sps: samples per symbol (default: 8)
        % h: Modulation index that is the product of frequency seperation and symbol duration (h>=1, default is 1)
        % ini_phase: Carrier initial phase (default: 0)
        % phase_cont: Phase continuity ('discont' | 'cont' (default))
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % symbol_order: Symbol order ('gray' (default) | 'bin')

        Mod = Define_Structure({'Type','M','sps','h','ini_phase','phase_cont','dfcTs','symbol_order'},...
            {'fsk', 2, 8, 1, 0, 'cont', 0, 'gray'},...
            varargin);

        % Symbol Mapping
        if strcmpi(Mod.symbol_order,'gray') % Gray encode and decode
            Mod.Txsymbolmap = bin2gray_ME('fsk',Mod.M); % Transmitter
            Mod.Rxsymbolmap = gray2bin_ME('fsk',Mod.M); % Reciever
        else
            Mod.Txsymbolmap = (0:Mod.M-1);
            Mod.Rxsymbolmap = (0:Mod.M-1);
        end

    case 'psk'

        % M: Modulation order (2 (default), 4, 8, 16)
        % phaserot: phase rotation of the modulation in radians (default: 0)
        % sps: samples per symbol (default: 8)
        % ini_phase: Carrier initial phase (default: 0)
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % beta: Roll-off factor for root raised cosine pulse (default: 0.35)
        % symbol_order: Symbol order ('gray' (default) | 'bin')

        Mod = Define_Structure({'Type','M','phaserot','sps','ini_phase','dfcTs','beta','symbol_order'},...
            {'psk',2,0,8,0,0,0.35,'gray'},...
            varargin);

        % Symbol Mapping
        if strcmpi(Mod.symbol_order,'gray') % Gray encode and decode
            Mod.Txsymbolmap = bin2gray_ME('psk',Mod.M); % Transmitter
            Mod.Rxsymbolmap = gray2bin_ME('psk',Mod.M); % Reciever
        else
            Mod.Txsymbolmap = (0:Mod.M-1);
            Mod.Rxsymbolmap = (0:Mod.M-1);
        end

        % Pulse Shaping
        sps_int = ceil(Mod.sps);
        Mod.PulseShaping.h = rcosdesign_ME(Mod.beta,8,sps_int,'sqrt'); % pulse shaping filter
        Mod.PulseShaping.delay = (length(Mod.PulseShaping.h)-1)/2; % filter delay (in samples)
        Mod.PulseShaping.half_symbol = round(sps_int/2); % half symbol time (in samples)

    case 'oqpsk'

        % M: Modulation order (fixed value 4)
        % phaserot: phase rotation of the modulation in radians (default: 0)
        % sps: samples per symbol (default: 8)
        % ini_phase: Carrier initial phase (default: 0)
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % beta: Roll-off factor for root raised cosine pulse (default: 0.35)
        % symbol_order: Symbol order ('gray' (default) | 'bin')

        Mod = Define_Structure({'Type','M','phaserot','sps','ini_phase','dfcTs','beta','symbol_order'},...
            {'oqpsk',4,0,8,0,0,0.35,'gray'},...
            varargin);

        % Symbol Mapping
        if strcmpi(Mod.symbol_order,'gray') % Gray encode and decode
            Mod.Txsymbolmap = bin2gray_ME('oqpsk',Mod.M); % Transmitter
            Mod.Rxsymbolmap = gray2bin_ME('oqpsk',Mod.M); % Reciever
        else
            Mod.Txsymbolmap = (0:Mod.M-1);
            Mod.Rxsymbolmap = (0:Mod.M-1);
        end

        % Pulse Shaping
        sps_int = ceil(Mod.sps);
        Mod.PulseShaping.h = rcosdesign_ME(Mod.beta,8,sps_int,'sqrt'); % pulse shaping filter
        Mod.PulseShaping.delay = (length(Mod.PulseShaping.h)-1)/2; % filter delay (in samples)
        Mod.PulseShaping.half_symbol = round(sps_int/2); % half symbol time (in samples)

    case 'qam'

        % M: Modulation order (8, 16 (default), 32, 64, 128)
        % phaserot: phase rotation of the modulation in radians (default: 0)
        % sps: samples per symbol (default: 8)
        % ini_phase: Carrier initial phase (default: 0)
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % beta: Roll-off factor for root raised cosine pulse (default: 0.35)
        % symbol_order: Symbol order ('gray' (default) | 'bin')

        Mod = Define_Structure({'Type','M','phaserot','sps','ini_phase','dfcTs','beta','symbol_order'},...
            {'qam',16,0,8,0,0,0.35,'gray'},...
            varargin);

        % Symbol Mapping
        if strcmpi(Mod.symbol_order,'gray') % Gray encode and decode
            Mod.Txsymbolmap = bin2gray_ME('qam',Mod.M); % Transmitter
            Mod.Rxsymbolmap = gray2bin_ME('qam',Mod.M); % Reciever
        else
            Mod.Txsymbolmap = (0:Mod.M-1);
            Mod.Rxsymbolmap = (0:Mod.M-1);
        end

        % Pulse Shaping
        sps_int = ceil(Mod.sps);
        Mod.PulseShaping.h = rcosdesign_ME(Mod.beta,8,sps_int,'sqrt'); % pulse shaping filter
        Mod.PulseShaping.delay = (length(Mod.PulseShaping.h)-1)/2; % filter delay (in samples)
        Mod.PulseShaping.half_symbol = round(sps_int/2); % half symbol time (in samples)

    case 'ask'

        % M: Modulation order (4(default), 8, 16)
        % phaserot: phase rotation of the modulation in radians (default: 0)
        % sps: samples per symbol (default: 8)
        % ini_phase: Carrier initial phase (default: 0)
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % beta: Roll-off factor for root raised cosine pulse (default: 0.35)
        % symbol_order: Symbol order ('gray' (default) | 'bin')

        Mod = Define_Structure({'Type','M','phaserot','sps','ini_phase','dfcTs','beta','symbol_order'},...
            {'ask',4,0,8,0,0,0.35,'gray'},...
            varargin);

        % Symbol Mapping
        if strcmpi(Mod.symbol_order,'gray') % Gray encode and decode
            Mod.Txsymbolmap = bin2gray_ME('ask',Mod.M); % Transmitter
            Mod.Rxsymbolmap = gray2bin_ME('ask',Mod.M); % Reciever
        else
            Mod.Txsymbolmap = (0:Mod.M-1);
            Mod.Rxsymbolmap = (0:Mod.M-1);
        end

        % Pulse Shaping
        sps_int = ceil(Mod.sps);
        Mod.PulseShaping.h = rcosdesign_ME(Mod.beta,8,sps_int,'sqrt'); % pulse shaping filter
        Mod.PulseShaping.delay = (length(Mod.PulseShaping.h)-1)/2; % filter delay (in samples)
        Mod.PulseShaping.half_symbol = round(sps_int/2); % half symbol time (in samples)
 
    case 'apsk'

        % M: 调制阶数 (16 (default), 32)
        % phaserot: 相位旋转 (弧度, 默认: 0)
        % sps: 每个符号的采样点数 (默认: 8)
        % ini_phase: 载波初始相位 (默认: 0)
        % dfcTs: 载波频偏与符号周期的乘积 (默认: 0)
        % beta: 滚降系数 (默认: 0.35)
        % symbol_order: 符号顺序 ('gray' | 'bin' (default))

        Mod = Define_Structure({'Type','M','phaserot','sps','ini_phase','dfcTs','beta','symbol_order'},...
            {'apsk',16,0,8,0,0,0.35,'bin'},... % 默认使用二进制映射
            varargin);

        % 对于生成模拟IQ数据的目的，二进制映射已足够
        if strcmpi(Mod.symbol_order,'gray')
            warning('标准APSK的格雷码映射复杂，当前版本将使用二进制映射替代。');
        end
        % 统一使用二进制映射
        Mod.Txsymbolmap = (0:Mod.M-1);
        Mod.Rxsymbolmap = (0:Mod.M-1);
        
        % Pulse Shaping (与QAM/ASK完全相同)
        sps_int = ceil(Mod.sps);
        Mod.PulseShaping.h = rcosdesign(Mod.beta,8,sps_int,'sqrt'); % 使用内建rcosdesign
        Mod.PulseShaping.delay = (length(Mod.PulseShaping.h)-1)/2;
        Mod.PulseShaping.half_symbol = round(sps_int/2);

    case 'am'

        % sps: Number of output signal samples corresponding to each input sample (i.e. fs/Fs; default: 8)
        % ini_phase: Carrier initial phase in radians (default: 0)
        % dfc_div_fs: Carrier frequency deviation of modulator divided by sampling rate (default: 0)
        % m: Modulation index that is peak value of message divided by carrier amplitude (default: 0.5)
        %   Note: if m>1, then overmodulation occurs.

        Mod = Define_Structure({'Type','sps','ini_phase','dfc_div_fs','m'},...
            {'am',8,0,0,0.5},...
            varargin);

    case 'dsb'

        % sps: Number of output signal samples corresponding to each input sample (i.e. fs/Fs; default: 8)
        % ini_phase: Carrier initial phase in radians (default: 0)
        % dfc_div_fs: Carrier frequency deviation of modulator divided by sampling rate (default: 0)

        Mod = Define_Structure({'Type','sps','ini_phase','dfc_div_fs',},...
            {'dsb',8,0,0},...
            varargin);

    case 'fm'

        % h: Frequency deviation of modulator (denoted by Df) divided by input signal maximum frequency (denoted by fm) (default: 4)
        %   Note 1: h is called modulation index.
        %   Note 2: Input signal maximum frequency is assumed to be equal to half
        %       the sampling rate of the provided input audio signal (denoted by Fs); i.e. fm = Fs/2.
        %   Note 3: Baseband FM bandwidth is equal to Df+fm = (1+h)*Fs/2, so the
        %       minimum sampling frequency should be fs>=(1+h)*Fs.
        % sps: Number of output signal samples corresponding to each input sample (i.e. fs/Fs; default: 8)
        % ini_phase: Carrier initial phase in radians (default: 0)
        % dfc_div_fs: Carrier frequency deviation of modulator divided by output signal sampling rate (default: 0)
        %   Note 4: When dfc_div_fs~=0, minimum sampling frequency should be fs>=(1+h)/(1-2*dfc_div_fs)*Fs.

        Mod = Define_Structure({'Type','h','sps','ini_phase','dfc_div_fs',},...
            {'fm',4,8,0,0},...
            varargin);

    case 'msk'
        % MSK (Minimum Shift Keying) is a special case of FSK.
        % It is a binary (M=2), continuous-phase FSK (CPFSK) with a
        % modulation index (h) of exactly 0.5.

        % M: Modulation order (fixed to 2 for MSK)
        % h: Modulation index (fixed to 0.5 for MSK)
        % sps: samples per symbol (default: 8)
        % ini_phase: Carrier initial phase (default: 0)
        % phase_cont: Phase cont inuity (fixed to 'cont' for MSK)
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % symbol_order: Symbol order (fixed to 'bin' for M=2)
        
        % 调用FSK的构造器，但强制M=2和h=0.5
        Mod = Define_Structure({'Type','M','sps','h','ini_phase','phase_cont','dfcTs','symbol_order'},...
             {'msk', 2, 8, 0.5, 0, 'cont', 0, 'bin'},...
              varargin);

        Mod.Type = 'msk';
        Mod.M = 2;
        Mod.h = 0.5;
        Mod.phase_cont = 'cont';
        Mod.symbol_order = 'bin';

        Mod.Txsymbolmap = [0, 1];
        Mod.Rxsymbolmap = [0, 1];

     case 'pi4qpsk' % ==================== 新增 case ====================
        % M: Modulation order (fixed to 4)
        % sps: samples per symbol (default: 8)
        % ini_phase: Carrier initial phase (default: 0)
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % beta: Roll-off factor for root raised cosine pulse (default: 0.35)
        % symbol_order: Symbol order ('gray' (default) | 'bin')

        Mod = Define_Structure({'Type','M','sps','ini_phase','dfcTs','beta','symbol_order'},...
            {'pi4qpsk',4,8,0,0,0.35,'gray'},...
            varargin);

        % 强制关键参数
        Mod.M = 4;
        Mod.Type = 'pi4qpsk';

        % 定义两个交替使用的星座图的绝对相位
        if strcmpi(Mod.symbol_order,'gray')
            % Gray mapping: 00->idx 1, 01->idx 2, 11->idx 3, 10->idx 4
            % Constellation 1 (standard QPSK): 0, pi/2, pi, 3pi/2
            Mod.Constellation1 = [0, pi/2, pi, 3*pi/2];
            % Constellation 2 (rotated by pi/4)
            Mod.Constellation2 = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4];
        else % Binary mapping: 00, 01, 10, 11
            Mod.Constellation1 = [0, pi/2, 3*pi/2, pi];
            Mod.Constellation2 = [pi/4, 3*pi/4, 7*pi/4, 5*pi/4];
        end

        sps_int = ceil(Mod.sps);
        Mod.PulseShaping.h = rcosdesign_ME(Mod.beta,8,sps_int,'sqrt');
        Mod.PulseShaping.delay = (length(Mod.PulseShaping.h)-1)/2;
        Mod.PulseShaping.half_symbol = round(sps_int/2);

    case 'pi4dqpsk'
        %M: Modulation order (fixed to 4 for DQPSK)
        % phaserot: phase rotation of the modulation in radians (default: 0)
        % sps: samples per symbol (default: 8)
        % ini_phase: Carrier initial phase (default: 0)
        % dfcTs: Product of carrier frequency deviation of modulator and symbol duration (default: 0)
        % beta: Roll-off factor for root raised cosine pulse (default: 0.35)
        % symbol_order: Symbol order ('gray' (default) | 'bin') - Gray is standard

        Mod = Define_Structure({'Type','M','phaserot','sps','ini_phase','dfcTs','beta','symbol_order'},...
            {'pi4dqpsk',4,0,8,0,0,0.35,'gray'},...
            varargin);

        % 强制关键参数
        Mod.M = 4; % DQPSK 阶数是4
        Mod.Type = 'pi4dqpsk';

        % π/4-DQPSK 不需要传统的符号映射，因为它是差分编码的。
        % 但我们可以保留它以保持结构一致性。
        if strcmpi(Mod.symbol_order,'gray')
            % 标准的 dibit-to-phase-change 映射
            % 00 -> +π/4, 01 -> +3π/4, 11 -> -3π/4, 10 -> -π/4
            Mod.PhaseMap = [pi/4, 3*pi/4, -3*pi/4, -pi/4];
        else % 二进制顺序
            % 00 -> +π/4, 01 -> +3π/4, 10 -> -π/4, 11 -> -3π/4
            Mod.PhaseMap = [pi/4, 3*pi/4, -pi/4, -3*pi/4];
        end

        sps_int = ceil(Mod.sps);
        Mod.PulseShaping.h = rcosdesign_ME(Mod.beta,8,sps_int,'sqrt');
        Mod.PulseShaping.delay = (length(Mod.PulseShaping.h)-1)/2;
        Mod.PulseShaping.half_symbol = round(sps_int/2);

end