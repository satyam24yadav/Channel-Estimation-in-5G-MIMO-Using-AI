## Methodology
This project implements and compares three channel estimation techniques for a 2×2 5G MIMO system:

1. Least Squares (LS) Channel Estimation
2. Minimum Mean Square Error (MMSE) Channel Estimation
3. Artificial Neural Network (ANN) based Channel Estimation

The system uses QPSK modulation over a Rayleigh fading channel with AWGN noise.
Performance is evaluated using Bit Error Rate (BER) versus Signal-to-Noise Ratio (SNR).

## Code Description
- ANN_Channel_Estimation.m : ANN-based channel estimation implementation
- LS_MMSE_Channel_Estimation.m : Conventional LS and MMSE estimators
- ANN_BER_vs_SNR.m : BER vs SNR performance evaluation

## Results
Simulation results show that ANN-based channel estimation achieves lower BER compared to LS and MMSE methods, especially at low SNR values.

## Tools Used
- MATLAB
- Neural Network Toolbox
