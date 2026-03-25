%% 5G MIMO Channel Estimation - Part 1
clc; clear; close all;

%% Step 0: Parameters
N = 1000;        % Total number of bits
Nt = 2;          % Number of transmit antennas
Nr = 2;          % Number of receive antennas
M = 4;           % QPSK modulation
SNR_dB = 0:5:30; % SNR range in dB

%% Step 1: Generate random bits
bits = randi([0 1], N, 1);

%% Step 2: Map bits to QPSK symbols
symbols = pskmod(bits, M, pi/4);

%% Step 3: Reshape symbols for MIMO
numSymbolsPerAntenna = length(symbols)/Nt;
X = reshape(symbols, Nt, numSymbolsPerAntenna);  % Nt x (numSymbols/Nt)

%% Step 4: Define Rayleigh MIMO channel
H = (randn(Nr,Nt) + 1i*randn(Nr,Nt))/sqrt(2);    % Nr x Nt

%% Step 5: Initialize arrays to store BER
BER_LS_arr = zeros(size(SNR_dB));
BER_MMSE_arr = zeros(size(SNR_dB));

%% Step 6: Loop over SNR values
for i = 1:length(SNR_dB)
    % Add AWGN noise
    snr = SNR_dB(i);
    noiseVar = 10^(-snr/10);
    noise = sqrt(noiseVar/2)*(randn(Nr,size(X,2)) + 1i*randn(Nr,size(X,2)));

    % Received signal
    Y = H*X + noise;  % Nr x numSymbolsPerAntenna

    % LS Channel Estimation
    H_LS = Y * pinv(X);  % Nr x Nt

    % MMSE Channel Estimation
    R_H = eye(Nt);       % Nt x Nt covariance
    sigma2 = noiseVar;
    H_MMSE = Y * X' * inv(X*X' + sigma2*eye(Nt)); % Nr x Nt

    %% Step 7: Equalization & Demodulation
    % LS Equalization
    eq_LS = pinv(H_LS)*Y;
    bits_hat_LS = pskdemod(eq_LS(:), M, pi/4);

    % MMSE Equalization
    eq_MMSE = pinv(H_MMSE)*Y;
    bits_hat_MMSE = pskdemod(eq_MMSE(:), M, pi/4);

    %% Step 8: BER Calculation
    BER_LS_arr(i) = sum(bits ~= bits_hat_LS)/N;
    BER_MMSE_arr(i) = sum(bits ~= bits_hat_MMSE)/N;
end

%% Step 9: Plot BER vs SNR
figure;
semilogy(SNR_dB, BER_LS_arr, '-o','LineWidth',2); hold on;
semilogy(SNR_dB, BER_MMSE_arr, '-x','LineWidth',2);
xlabel('SNR (dB)'); ylabel('Bit Error Rate (BER)');
title('BER vs SNR for 5G MIMO - LS vs MMSE');
legend('LS','MMSE'); grid on;

%% Step 10: Display final estimated channels (for verification)
disp('Actual Channel H:'); disp(H);
disp('LS Estimated Channel H_LS:'); disp(H_LS);
disp('MMSE Estimated Channel H_MMSE:'); disp(H_MMSE);

%% Step 11: Save results
save('Part1_Results.mat','BER_LS_arr','BER_MMSE_arr','H','H_LS','H_MMSE');
saveas(gcf,'BER_vs_SNR.png');
saveas(gcf,'BER_vs_SNR.fig');
