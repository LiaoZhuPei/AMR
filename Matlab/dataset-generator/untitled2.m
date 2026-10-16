% --- 診斷腳本 ---
clc;
clear;

% 1. 調用您的星座圖生成函數
M = 32;
constellation_output = apsk_constellation_NN(M);

% 2. 打印出所有獨立的、不重複的符號值
unique_symbols = unique(constellation_output);
disp('星座圖中所有獨立的符號值:');
disp(unique_symbols);

% 3. 繪製星座圖
figure;
scatter(real(constellation_output), imag(constellation_output), 100, 'b', 'filled');
grid on;
axis square;
title(sprintf('由 apsk_constellation_NN(%d) 生成的星座圖', M));
xlabel('I (同相分量)');
ylabel('Q (正交分量)');