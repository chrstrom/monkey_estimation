#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import cfg

SNRs = [-10, 0, 10, 20, 30, 40, 50, 60]
Ks = [10, 12, 14, 16, 18, 20]

N = len(SNRs)
M = len(Ks)

crlb_w = np.empty(N*M)
crlb_phi = np.empty(N*M)

var_w = np.empty(N*M)
var_phi = np.empty(N*M)

mean_w = np.empty(N*M)
mean_phi = np.empty(N*M)

w_estimate_valid = np.empty(N*M)
phi_estimate_valid = np.empty(N*M)


# format: SNR_dB, K, crlb_w, var_w, w_estimate_valid, crlb_phi, var_phi, phi_estimate_valid, mean_w, mean_phi
with open('./data/run_2_N_30000.csv') as csvfile:

    reader = csv.reader(csvfile, delimiter=' ')

    i = 0
    for row in reader:
        K           = row[1]

        crlb_w[i]   = row[2]
        var_w[i]    = row[3]

        crlb_phi[i] = row[5]
        var_phi[i]  = row[6]

        mean_w[i]   = row[8]
        mean_phi[i] = row[9]

        w_estimate_valid[i]   = bool(row[4])
        phi_estimate_valid[i] = bool(row[7])

        i += 1

N_ROWS = M/2
N_COLS = 2

w_fig, ax = plt.subplots(N_ROWS, N_COLS)
plt.figure(w_fig.number)
plt.title("Variance for the omega estimate for varying FFT length")
plt.tight_layout()
for i in range(N_ROWS):
    for j in range(N_COLS):
        n = range(N*(2*i+j), N*(2*i+j+1))
        ax[i][j].semilogy(SNRs, crlb_w[n], 'k.:')
        ax[i][j].semilogy(SNRs, var_w[n], 'r.-')
        ax[i][j].set_title("FFT length = 2^" + str(4*i + 2*j + 10))
        ax[i][j].set_xlabel("SNR")
        ax[i][j].set_xlabel("Variance")
        ax[i][j].legend(['CRLB', 'Estimator'])


phi_fig, ax = plt.subplots(N_ROWS, N_COLS)
plt.figure(phi_fig.number)
plt.title("Variance for the phi estimate for varying FFT length")
plt.tight_layout()
for i in range(N_ROWS):
    for j in range(N_COLS):
        n = range(N*(2*i+j), N*(2*i+j+1))
        ax[i][j].semilogy(SNRs, crlb_phi[n], 'k.:')
        ax[i][j].semilogy(SNRs, var_phi[n], 'r.-')
        ax[i][j].set_title("FFT length = 2^" + str(4*i + 2*j + 10))
        ax[i][j].set_xlabel("SNR")
        ax[i][j].set_xlabel("Variance")
        ax[i][j].legend(['CRLB', 'Estimator'])


plt.figure(3)
#plt.title("Difference in the mean omega from one SNR value to the next, for varying FFT length")
plt.title("Mean for the omega estimate for varying FFT length and SNRs")
plt.plot([-10, 60], [cfg.w0, cfg.w0], 'k')
for i in range(M):
    n = range(N*i, N*(i+1))
    #plt.semilogy(SNRs[1:], np.abs(np.diff(mean_w[n])), '.--')
    plt.plot(SNRs, mean_w[n], '.--')

plt.legend(["True value", "2^10", "2^12", "2^14", "2^16", "2^18", "2^20"])
plt.xlabel("SNR")
plt.ylabel("Mean")



plt.figure(4)
plt.title("Mean for the phi estimate for varying FFT length and SNRs")
plt.plot([-10, 60], [cfg.phi, cfg.phi], 'k')
for i in range(M):
    n = range(N*i, N*(i+1))
    plt.plot(SNRs, mean_phi[n], '.--')

plt.legend(["True value", "2^10", "2^12", "2^14", "2^16", "2^18", "2^20"])
plt.xlabel("SNR")
plt.ylabel("Mean")

plt.show()
