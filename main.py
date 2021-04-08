#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scripts import signals as sig
from scripts import fft_estimator
from scripts import utility
from scripts import crlb
from scripts import cfg

SNR_dBs = [-10, 0, 10, 20]#, 30, 40, 50, 60]
FFT_Ks = [10, 12, 14, 16, 18, 20] # Commented out for performance boost when testing

n = len(SNR_dBs)
m = len(FFT_Ks)
N = 100 # Amount of samples to generate when estimating variance


for i in range(m):
    K = FFT_Ks[i]
    M = 2**K

    for j in range(n):
        SNR = sig.linear_SNR_from_dB(SNR_dBs[j])

        w_estimates = np.zeros(N)
        phi_estimates = np.zeros(N)

        status_bar_progress = 0
        for k in range(N):
            x_d = sig.generate_signal(SNR)

            omega_hat, phi_hat, _ = fft_estimator.estimator(x_d, M)

            w_estimates[k] = omega_hat
            phi_estimates[k] = phi_hat

            status_bar_progress = utility.print_status_bar(k, status_bar_progress, N)

        mean_w = np.mean(w_estimates)
        mean_phi = np.mean(phi_estimates)

        var_w = np.var(w_estimates)
        var_phi = np.var(phi_estimates)

        crlb_w = crlb.omega(SNR)
        crlb_phi = crlb.phi(SNR)

        print("") # Newline
        print("CONFIG | SNR [dB]: {}, M: 2^{}, true omega: {}, true phase: {}".format(SNR_dBs[j], K, cfg.w0, cfg.phi))
        print("OMEGA  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_w, var_w, crlb_w))
        print("PHASE  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_phi, var_phi, crlb_phi))
        print("") # Newline