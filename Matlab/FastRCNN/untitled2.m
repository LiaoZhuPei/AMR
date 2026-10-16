m = matfile('IQREC.mat');
[rows, cols] = size(m, 'iq_complex');
part_data = m.iq_complex(1, 100000:200000);