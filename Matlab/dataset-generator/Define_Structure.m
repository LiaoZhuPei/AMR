
% Inputs:
%   default_fields: 1xN cell array of strings denoting the field names
%   default_values: 1xN cell array denoting the field values
%   fields_values_pairs: Row cell array with length 2xL that can be used to assign non-default values to some fields
%       For i=1,2,3,...,L, fields_values_pairs{2*i-1} is the desired field name and
%       fields_values_pairs{2*i} is the desired field value.
%
% Output:
%   S: Defined structure.

function S = Define_Structure(default_fields,default_values,fields_values_pairs)

% Assign default values
N = length(default_fields);
for j=1:N
    S.(default_fields{j}) = default_values{j};
end

% Change Default Values Indicated by fields_values_pairs
L = length(fields_values_pairs)/2;
if L==0
    return;
end
fields_values_pairs = reshape(fields_values_pairs,2,[]);
for i=1:L
    if sum(cellfun(@strcmp,default_fields,repmat(fields_values_pairs(1,i),1,length(default_fields))))~=1
        error('Error in using Define_Structure_ME');
    end
    
    S.(fields_values_pairs{1,i}) = fields_values_pairs{2,i};
end