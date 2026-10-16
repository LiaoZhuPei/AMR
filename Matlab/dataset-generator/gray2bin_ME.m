
% Inputs:
%   ModType: Modulation type ('fsk', 'psk', 'dpsk', 'oqpsk', 'qam', 'ask')
%   M: Modulation order
%
% Output:
%   MAP: 1xM vector of constellation map for demodulation


function MAP = gray2bin_ME(ModType,M)

switch lower(ModType)

    case {'fsk','psk','dpsk','oqpsk','ask'}

        % Calculate map
        j = (0:M-1);
        mapping = bitxor(j,bitshift(j,-1));

        % Format output and translate x (map) i.e. Gray Decode
        x = (0:M-1);
        [~,index] = ismember(x,mapping);
        MAP = index-1;

    case {'qam'}

        k = log2(M);                % Number of bits per symbol
        mapping = (0:M-1)';         % Binary mapping to be Gray converted
        if rem(k,2) % non-square constellation

            kI = (k+1)/2;
            kQ = (k-1)/2;

            symbolI = bitshift(mapping,-kQ);
            symbolQ = bitand(mapping,bitshift(M-1,-kI));

            % while i is smaller (Number of bits per symbol)/2
            i = 1;
            while i < kI
                tempI = symbolI;
                tempI = bitshift(tempI,-i);
                symbolI = bitxor(symbolI,tempI);
                i = i + i;  % i takes on values 1,2,4,8,...,2^n - n is an integer
            end

            % while i is smaller (Number of bits per symbol)/2
            i = 1;
            while i < kQ
                tempQ = symbolQ;
                tempQ = bitshift(tempQ,-i);
                symbolQ = bitxor(symbolQ, tempQ);
                i = i + i;  % i takes on values 1,2,4,8,...,2^n - n is an integer
            end

            SymbolIndex = double(bitshift(symbolI,kQ) + symbolQ);

        else % square constellation

            symbolI = bitshift(mapping,-k/2);
            symbolQ = bitand(mapping,bitshift(M-1,-k/2));

            % while i is smaller (Number of bits per symbol)/2
            i = 1;
            while i < k/2
                tempI = symbolI;
                tempI = bitshift(tempI,-i);
                symbolI = bitxor(symbolI,tempI);

                tempQ = symbolQ;
                tempQ = bitshift(tempQ,-i);
                symbolQ = bitxor(symbolQ, tempQ);
                i = i + i;   % i takes on values 1,2,4,8,...,2^n - n is an integer
            end

            SymbolIndex = double(bitshift(symbolI,k/2) + symbolQ);

        end

        mapping = SymbolIndex;

        % Make sure that mapping is a vector, when used to name the symbols
        % column-wise starting from left upper corner, results in a gray mapped
        % constellation.
        [~,index]=ismember(0:M-1,mapping);
        mapping = index - 1;
        
        
        % We can use this new mapping to decode Gray encoding
        % Format output and translate x (map) i.e. Gray decode
        MAP = mapping(1:M);


end