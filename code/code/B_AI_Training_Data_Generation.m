%% 5G MIMO Channel Estimation – Part 1 + AI Data Generation
clc; clear; close all;

%% ---------------- Step 0: System Parameters ----------------
N = 1000;          % Total number of bits
Nt = 2;            % Number of transmit antennas
Nr = 2;            % Number of receive antennas
M = 4;             % QPSK modulation
SNR_dB = 10;       % Fixed SNR for AI data generation
numSamples = 2000; % Number of training samples for ANN/CNN

%% WHY:
% Nt=Nr=2 → 2x2 MIMO (simple + clear)
% numSamples → ANN/CNN ko seekhne ke liye multiple examples chahiye

%% ---------------- Step 1: Generate Random Bits ----------------
bits = randi([0 1], N, 1);

%% WHY:
% Ye transmitted information hai jo system me bheja jaata hai

%% ---------------- Step 2: QPSK Modulation ----------------
symbols = pskmod(bits, M, pi/4);

%% WHY:
% QPSK 5G systems me commonly used modulation hai
% Complex symbols generate hote hain (real + imag)

%% ---------------- Step 3: MIMO Symbol Mapping ----------------
numSymbolsPerAntenna = length(symbols)/Nt;
X = reshape(symbols, Nt, numSymbolsPerAntenna);

%% WHY:
% Symbols ko 2 transmit antennas me divide kiya
% X = transmitted signal matrix

%% ---------------- Step 4: Preallocate Storage (IMPORTANT) ----------------
Y_data = zeros(Nr, size(X,2), numSamples);
H_data = zeros(Nr, Nt, numSamples);

%% WHY:
% ANN/CNN ke liye:
% Input  = Received signal Y
% Output = Actual channel H
% Isliye dono ko store karna zaroori hai

%% ---------------- Step 5: Loop for Data Generation ----------------
for k = 1:numSamples

    % ----- Rayleigh Channel -----
    H = (randn(Nr,Nt) + 1i*randn(Nr,Nt)) / sqrt(2);

    % WHY:
    % Rayleigh fading real wireless multipath ko represent karta hai

    % ----- AWGN Noise -----
    noiseVar = 10^(-SNR_dB/10);
    noise = sqrt(noiseVar/2) * ...
            (randn(Nr,size(X,2)) + 1i*randn(Nr,size(X,2)));

    % WHY:
    % Noise real channel conditions ko simulate karta hai

    % ----- Received Signal -----
    Y = H * X + noise;

    % WHY:
    % Actual communication equation: Y = H·X + N

    % ----- Store for AI Training -----
    Y_data(:,:,k) = Y;
    H_data(:,:,k) = H;

end

disp('AI training data generated successfully');

%% ---------------- Step 6: Verify One Sample using LS/MMSE ----------------

% Take one sample
Y_test = Y_data(:,:,1);
H_actual = H_data(:,:,1);

% ----- LS Estimation -----
H_LS = Y_test * pinv(X);

% ----- MMSE Estimation -----
sigma2 = noiseVar;
H_MMSE = Y_test * X' * inv(X*X' + sigma2*eye(Nt));

%% WHY:
% LS/MMSE ko yaha verify kar rahe hain
% Taaki proof mile ki generated data correct hai

%% ---------------- Step 7: Display Results ----------------
disp('Actual Channel H:');
disp(H_actual);

disp('LS Estimated Channel H_LS:');
disp(H_LS);

disp('MMSE Estimated Channel H_MMSE:');
disp(H_MMSE);

%% ---------------- Step 8: Save Data ----------------
save('AI_Training_Data.mat','Y_data','H_data','X');

%% WHY:
% Ye file ANN/CNN training me directly use hogi
