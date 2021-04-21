#!/usr/bin/env python

import csv
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

from scripts import signals as sig
from scripts import fft_estimator
from scripts import utility
from scripts import crlb
from scripts import cfg

SNR_dBs = [-10, 0, 10, 20, 30, 40, 50, 60]
FFT_Ks = [10, 12, 14, 16, 18, 20] # Commented out for performance boost when testing

n = len(SNR_dBs)
m = len(FFT_Ks)
N = 1000 # Amount of samples to generate when estimating variance

run_number = len([name for name in os.listdir('./data') if os.path.isfile('./data/' + name)])
filename = 'data/run_' + str(run_number) + '_N_' + str(N) + '.csv'

with open(filename, 'ab') as file:
    writer = csv.writer(file, delimiter=' ')

    total_time_begin = dt.now()
    for i in range(m):
        K = FFT_Ks[i]
        M = 2**K

        for j in range(n):
            SNR_dB = SNR_dBs[j]

            w_estimates = np.zeros(N)
            phi_estimates = np.zeros(N)

            status_bar_progress = 0
            run_time_begin = dt.now()
            for k in range(N):
                x_d = sig.x_discrete(SNR_dB)

                omega_hat, phi_hat, _, _ = fft_estimator.estimator(x_d, M)

                w_estimates[k] = omega_hat
                phi_estimates[k] = phi_hat

                status_bar_progress = utility.print_status_bar(k, status_bar_progress, N)

            mean_w = np.mean(w_estimates) / m
            mean_phi = np.mean(phi_estimates)

            var_w = np.var(w_estimates)
            var_phi = np.var(phi_estimates)

            crlb_w = crlb.omega(SNR_dB)
            crlb_phi = crlb.phi(SNR_dB)

            run_time_end = dt.now()
            print("")
            utility.print_execution_time(run_time_begin, run_time_end)

            w_estimate_valid = True
            phi_estimate_valid = True
            if var_w < crlb_w:
                w_estimate_valid = False
                print("Variance for omega lower than CRLB!")

            if var_phi < crlb_phi:
                phi_estimate_valid = False
                print("Variance for phi lower than CRLB!")
                

            writer.writerow([SNR_dB, K, crlb_w, var_w, w_estimate_valid, crlb_phi, var_phi, phi_estimate_valid, mean_w, mean_phi])

            # print("CONFIG | SNR [dB]: {}, M: 2^{}, true omega: {}, true phase: {}".format(SNR_dB, K, cfg.w0, cfg.phi))
            # print("OMEGA  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_w, var_w, crlb_w))
            # print("PHASE  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_phi, var_phi, crlb_phi))
            # print("")

    total_time_end = dt.now()
    utility.print_execution_time(total_time_begin, total_time_end)
