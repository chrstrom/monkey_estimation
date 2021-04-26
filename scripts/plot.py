#!/usr/bin/env python

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import cfg


def plot_var_task_a(ax, crlb, estimator_variance, ylim=None):
    plt.tight_layout()
    for i in range(N_ROWS):
        for j in range(N_COLS):
            n = range(N*(2*i+j), N*(2*i+j+1))
            axis = ax[i][j]
            axis.semilogy(SNRs, crlb[n], 'k.:')
            axis.semilogy(SNRs, estimator_variance[n], 'r.-')
            axis.set_title("FFT length = 2^" + str(4*i + 2*j + 10))
            axis.set_xlabel("SNR")
            axis.set_ylabel("Variance")
            axis.legend(['CRLB', 'Estimator'])

            if ylim is not None:
                axis.set_ylim(ylim)

def plot_mean_task_a(true_mean, estimated_mean):
    plt.tight_layout()
    plt.plot([-10, 60], [true_mean, true_mean], 'k')
    for i in range(M):
        n = range(N*i, N*(i+1))
        plt.plot(SNRs, estimated_mean[n], '.--')

    plt.legend(["True value", "2^10", "2^12", "2^14", "2^16", "2^18", "2^20"])
    plt.xlabel("SNR")
    plt.ylabel("Mean")

def plot_var_task_b(crlb, estimator_variance):
    plt.tight_layout()
    plt.semilogy(SNRs, crlb[0:N], 'k.:')
    plt.semilogy(SNRs, estimator_variance[0:N], 'r.-')
    plt.legend(['CRLB', 'Estimator'])
    plt.xlabel("SNR")
    plt.ylabel("Variance")

def plot_mean_task_b(true_mean, estimated_mean):
    plt.rc('axes.formatter', useoffset=False)
    plt.plot([-10, 60], [true_mean, true_mean], 'k')
    plt.plot(SNRs, estimated_mean, 'r.--')
    plt.legend(["True value", "Fine-tuned estimate"])
    plt.xlabel("SNR")
    plt.ylabel("Mean")


if __name__ == '__main__':
    task = sys.argv[1]

    if task == 'a':
        filename = "./data/part_a_run_7_N_100.csv"
    elif task == 'b':
        filename= './data/part_b_run_6_N_1000.csv'
    else:
        print("'Task' argument has to either be 'a' or 'b', exiting...")
        exit(1)
        
    SNRs = [-10, 0, 10, 20, 30, 40, 50, 60]
    Ks = [10, 12, 14, 16, 18, 20]

    N = len(SNRs)
    M = len(Ks)

    crlb_w = np.empty(N*M)
    crlb_phi = np.empty(N*M)

    var_w = np.empty(N*M)
    var_phi = np.empty(N*M)

    mean_f = np.empty(N*M)
    mean_phi = np.empty(N*M)

    w_estimate_valid = np.empty(N*M)
    phi_estimate_valid = np.empty(N*M)


    with open(filename) as csvfile:

        reader = csv.reader(csvfile, delimiter=' ')

        i = 0
        for row in reader:
            K           = row[1]

            crlb_w[i]   = row[2]
            var_w[i]    = row[3]

            crlb_phi[i] = row[5]
            var_phi[i]  = row[6]

            mean_f[i]   = row[8]
            mean_phi[i] = row[9]

            w_estimate_valid[i]   = bool(row[4])
            phi_estimate_valid[i] = bool(row[7])

            i += 1

    if task == 'a':
        N_ROWS = M/2
        N_COLS = 2

        _, ax = plt.subplots(N_ROWS, N_COLS)
        plt.figure(1)
        plot_var_task_a(ax, crlb_w, var_w, [0.01, 1e7])

        _, ax = plt.subplots(N_ROWS, N_COLS)
        plt.figure(2)
        plot_var_task_a(ax, crlb_phi, var_phi)


        plt.figure(3)
        plot_mean_task_a(cfg.f0, mean_f)

        plt.figure(4)
        plot_mean_task_a(cfg.phi, mean_phi)

    if task == 'b':

        plt.figure(1)
        plot_var_task_b(crlb_w, var_w)

        plt.figure(2)
        plot_var_task_b(crlb_phi, var_phi)

        plt.figure(3)
        plot_mean_task_b(cfg.f0, mean_f[0:N])

        plt.figure(4)
        plot_mean_task_b(cfg.phi, mean_phi[0:N])

    plt.show()

