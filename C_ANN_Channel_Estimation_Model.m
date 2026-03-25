%% =====================================================
% ANN based Channel Estimation (5G 2x2 MIMO)
% =====================================================

clc; clear; close all;

%% ---------- STEP 1: Load AI Training Data ----------
load('AI_Training_Data.mat');     % Y_data, H_data, X
disp('Training data loaded');

numSamples = size(Y_data,3);
[Nr, Nt] = size(H_data(:,:,1));

%% ---------- STEP 2: Prepare ANN Input & Output ----------
X_train = [];
Y_train = [];

for k = 1:numSamples

    % ----- Input: Received Signal Y -----
    Yk = Y_data(:,:,k);
    Y_real = real(Yk(:));
    Y_imag = imag(Yk(:));
    X_train(:,k) = [Y_real; Y_imag];

    % ----- Output: Actual Channel H -----
    Hk = H_data(:,:,k);
    H_real = real(Hk(:));
    H_imag = imag(Hk(:));
    Y_train(:,k) = [H_real; H_imag];

end

% Transpose for ANN format
X_train = X_train';
Y_train = Y_train';

disp('ANN input-output data prepared');

%% ---------- STEP 3: Create ANN Model ----------
net = feedforwardnet([32 16]);   % simple & safe architecture

net.trainFcn = 'trainscg';       % memory efficient
net.performFcn = 'mse';

net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.1;
net.divideParam.testRatio  = 0.1;

disp('ANN model created');

%% ---------- STEP 4: Train ANN ----------
net = train(net, X_train', Y_train');
disp('ANN training completed');
save('trained_ANN.mat' , 'net');

%% ---------- STEP 5: Test ANN on One Sample ----------
X_test = X_train(1,:)';          % received signal
H_actual_vec = Y_train(1,:)';    % actual channel vector

H_ANN_vec = net(X_test);         % ANN prediction

%% ---------- STEP 6: Convert ANN Output to Channel Matrix ----------
len = length(H_ANN_vec)/2;
H_real = H_ANN_vec(1:len);
H_imag = H_ANN_vec(len+1:end);

H_ANN = reshape(H_real + 1i*H_imag, Nr, Nt);

%% ---------- STEP 7: Display Results ----------
disp('Actual Channel H:');
disp(H_data(:,:,1));

disp('ANN Estimated Channel H_ANN:');
disp(H_ANN);
