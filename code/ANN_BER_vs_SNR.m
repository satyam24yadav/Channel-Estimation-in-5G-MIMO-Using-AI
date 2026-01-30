clc; clear; close all;

%% ---------------- Load Trained ANN ----------------
load('AI_Training_Data.mat');   % Y_data, H_data, X
load('trained_ANN.mat');        % net (save this after training)

%% ---------------- System Parameters ----------------
N = 1000;
Nt = 2;
Nr = 2;
M = 4;
SNR_dB = 0:5:30;

BER_ANN = zeros(size(SNR_dB));

%% ---------------- Generate Random Bits ----------------
bits = randi([0 1], N, 1);
symbols = pskmod(bits, M, pi/4);

numSymbolsPerAntenna = length(symbols)/Nt;
X_tx = reshape(symbols, Nt, numSymbolsPerAntenna);

%% ---------------- SNR Loop ----------------
for i = 1:length(SNR_dB)

    % Channel
    H = (randn(Nr,Nt) + 1i*randn(Nr,Nt))/sqrt(2);

    % Noise
    snr = SNR_dB(i);
    noiseVar = 10^(-snr/10);
    noise = sqrt(noiseVar/2) * ...
        (randn(Nr,size(X_tx,2)) + 1i*randn(Nr,size(X_tx,2)));

    % Received Signal
    Y = H*X_tx + noise;

    %% -------- ANN Channel Estimation --------
    Y_real = real(Y(:));
    Y_imag = imag(Y(:));
    ANN_input = [Y_real; Y_imag];

    H_ann_vec = net(ANN_input);

    % Convert ANN output to complex channel
    H_real = H_ann_vec(1:Nr*Nt);
    H_imag = H_ann_vec(Nr*Nt+1:end);
    H_ANN = reshape(H_real + 1i*H_imag, Nr, Nt);

    %% -------- Equalization using ANN --------
    eq_ANN = pinv(H_ANN) * Y;

    %% -------- Demodulation --------
    bits_hat_ANN = pskdemod(eq_ANN(:), M, pi/4);

    %% -------- BER Calculation --------
    BER_ANN(i) = sum(bits ~= bits_hat_ANN) / N;

end

%% ---------------- Plot BER vs SNR ----------------
figure;
semilogy(SNR_dB, BER_ANN, '-o','LineWidth',2);
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs SNR using ANN-based Channel Estimation');
grid on;

disp('ANN BER vs SNR completed');
