#!/usr/bin/env python

import csv
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from scipy import optimize

from scripts import signals as sig
from scripts import fft_estimator
from scripts import optimizing
from scripts import utility
from scripts import crlb
from scripts import cfg

task = 'a'

SNR_dBs =[-10, 0, 10, 20, 30, 40, 50, 60]
FFT_Ks = [10, 12, 14, 16, 18, 20]

n = len(SNR_dBs)
m = len(FFT_Ks)
N = 1000  # Amount of samples to generate when estimating variance

# Generate unique filename for data file output
run_number = len([name for name in os.listdir('./data') if os.path.isfile('./data/' + name)])

if task == 'a':
    filename = 'data/part_a_run_' + str(run_number) + '_N_' + str(N) + '.csv'
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

                mean_f = np.mean(w_estimates) / (2*np.pi)
                mean_phi = np.mean(phi_estimates)

                var_f = np.var(w_estimates)
                var_phi = np.var(phi_estimates)

                crlb_f = crlb.omega(SNR_dB)
                crlb_phi = crlb.phi(SNR_dB)

                run_time_end = dt.now()
                print("")
                utility.print_execution_time(run_time_begin, run_time_end)

                f_estimate_valid = True
                phi_estimate_valid = True
                if var_f < crlb_f:
                    f_estimate_valid = False
                    print("Variance for frequency lower than CRLB!")

                if var_phi < crlb_phi:
                    phi_estimate_valid = False
                    print("Variance for phi lower than CRLB!")
                    

                writer.writerow([SNR_dB, K, crlb_f, var_f, f_estimate_valid, crlb_phi, var_phi, phi_estimate_valid, mean_f, mean_phi])

                print("CONFIG | SNR [dB]: {}, M: 2^{}, true frequency: {}, true phase: {}".format(SNR_dB, K, cfg.f0, cfg.phi))
                print("FREQUENCY  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_f, var_f, crlb_f))
                print("PHASE  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_phi, var_phi, crlb_phi))
                print("")

        total_time_end = dt.now()
        utility.print_execution_time(total_time_begin, total_time_end)

if task == 'b':
    filename = 'data/part_b_run_' + str(run_number) + '_N_' + str(N) + '.csv'
    with open(filename, 'ab') as file:
        writer = csv.writer(file, delimiter=' ')
        M = 2**10
        for SNR_dB in SNR_dBs:

            w_estimates = np.zeros(N)
            phi_estimates = np.zeros(N)

            status_bar_progress = 0
            run_time_begin = dt.now()

            for i in range(N):

                x_d = sig.x_discrete(SNR_dB)

                omega_hat, phi_hat, _, _ = fft_estimator.estimator(x_d, M)

                omega_opt = optimize.minimize(optimizing.frequency_objective_function, omega_hat, method="Nelder-Mead", args=(M, x_d, phi_hat))
                phase_opt = optimize.minimize(optimizing.phase_objective_function, phi_hat, method="Nelder-Mead", args=(x_d, omega_hat))

                w_estimates[i] = omega_opt.x[0]
                phi_estimates[i] = phase_opt.x[0]

                status_bar_progress = utility.print_status_bar(i, status_bar_progress, N)

            run_time_end = dt.now()
            print("")
            utility.print_execution_time(run_time_begin, run_time_end)

            mean_f = np.mean(w_estimates) / (2*np.pi)
            mean_phi = np.mean(phi_estimates)

            var_f = np.var(w_estimates)
            var_phi = np.var(phi_estimates)

            crlb_f = crlb.omega(SNR_dB)
            crlb_phi = crlb.phi(SNR_dB)

            f_estimate_valid = True
            phi_estimate_valid = True
            
            if var_f < crlb_f:
                f_estimate_valid = False
                print("Variance for f lower than CRLB!")

            if var_phi < crlb_phi:
                phi_estimate_valid = False
                print("Variance for phi lower than CRLB!")


            writer.writerow([SNR_dB, 10, crlb_f, var_f, f_estimate_valid, crlb_phi, var_phi, phi_estimate_valid, mean_f, mean_phi])
            print("CONFIG | SNR [dB]: {}, M: 2^{}, true f: {}, true phase: {}".format(SNR_dB, 10, cfg.f0, cfg.phi))
            print("FREQUENCY  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_f, var_f, crlb_f))
            print("PHASE  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_phi, var_phi, crlb_phi))
            print("")
