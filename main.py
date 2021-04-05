#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scripts import Signals
from scripts import fast_dtft
from scripts import cfg

import math

SNR_dBs = [-10, 0, 10, 20, 30, 40, 50, 60]
FFT_Ks = [10, 12, 14, 16, 18, 20]

n = len(SNR_dBs)
m = len(FFT_Ks)

num_plots_y = 2
num_plots_x = m / num_plots_y

for i in range(n):

    SNR = SNR_dBs[i]

    sig = Signals.Signals(SNR)
    x = sig.x_discrete()

    fig = plt.figure(i)
    for j in range(m):
        k = FFT_Ks[j]
        M = 2**k
        fft = fast_dtft.FastDTFT(M)
        Fx, Ff = fft.fast_dtft(x)

        plt.subplot(num_plots_y, num_plots_x, j+1)
        plt.plot(Ff, abs(Fx)/max(abs(Fx)), 'k')
        plt.title("2^%d-point FFT for x[n], SNR = %d dB" % (k, SNR))

        # Comparison between CRLB and sigma^2 goes here

    fig.text(0.5, 0.04, "Frequency [Hz]", ha='center')
    fig.text(0.04, 0.5, "abs(Fx)", va='center', rotation='vertical')

plt.show()
