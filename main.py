#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scripts import Signals
from scripts import estimation
from scripts import CRLB

from scipy import fft, ifft, fftpack

SNR_dBs = [-10, 0, 10, 20, 30, 40, 50, 60]
FFT_Ks = [10, 12, 14]#, 16, 18, 20] # Commented out for performance boost when testing

n = len(SNR_dBs)
m = len(FFT_Ks)
N = 100 # Amount of samples to generate when estimating variance

for i in range(m):
    K = FFT_Ks[i]
    M = 2**K

    M_point_estimator = estimation.FFTEstimator(M)

    for j in range(n):
        SNR = SNR_dBs[j]
        sig = Signals.Signals(SNR)
        crlb = CRLB.CRLB(SNR)

        omega_estimates = np.zeros(N)
        phase_estimates = np.zeros(N)
        for k in range(N):
            x_d = sig.x_discrete()

            omega_hat, phi_hat = M_point_estimator.estimate_omega_and_phi(x_d)

            omega_estimates[k] = omega_hat
            phase_estimates[k] = phi_hat

        var_omega = np.var(omega_estimates)
        var_phase = np.var(phase_estimates)

        print("Samples: {}, SNR: {}, M: 2^{}, var_omega: {}, CRLB: {}".format(N, SNR, K, var_omega, crlb.omega()))
        print("Samples: {}, SNR: {}, M: 2^{}, var_phase: {}, CRLB: {}".format(N, SNR, K, var_phase, crlb.phi()))

    print("") # Newline