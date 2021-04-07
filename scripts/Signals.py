#!/usr/bin/env python

from cmath import exp, pi
import numpy as np
import matplotlib.pyplot as plt

import cfg

def sigma_squared_from_SNR(A, SNR):
    return A**2 / (2*float(SNR))

class Signals:

    def __init__(self, SNR_dB=None):
        self.N   = cfg.N
        self.Fs  = cfg.Fs
        self.Ts  = cfg.Ts
        self.A   = cfg.A

        self.P   = cfg.P
        self.Q   = cfg.Q
        self.n0  = cfg.n0

        self.phi = cfg.phi

        self.f0  = cfg.f0
        self.w0  = cfg.w0

        if SNR_dB is None:
            self.SNR = cfg.SNR
        else:
            self.SNR = 10**(SNR_dB/10.0)

        self.sigma = sigma_squared_from_SNR(self.A, self.SNR)

    def F(self, w0):
        x = self.x_discrete()
        sum = 0
        for n in range(0, self.N):
            sum += x[n]*exp(-1j*w0*n*self.Ts)

        return sum / self.N

    def x_discrete(self):
        # Generate the data for the sampled signal
        x = [0 for i in range(self.N)]
        noise_real = np.random.normal(0, pow(self.sigma, 2), self.N)
        noise_imag = np.random.normal(0, pow(self.sigma, 2), self.N)

        n = self.n0
        for i in range(self.N):
            z = complex(0, self.w0 * n * self.Ts + self.phi)
            x[i] = self.A * exp(z) + complex(noise_real[i], noise_imag[i])
            n += 1

        return x

    def x_ang_frequency(self, ang_frequency):
        # Generates a theoretical signal without noise
        x = [0 for i in range(self.N)]
        
        n = self.n0
        for i in range(self.N):
            z = complex(0, self.w0 * n * self.Ts + self.phi)
            x[i] = self.A * exp(z)
            n += 1

        return x


    def x_phase(self, phase):
        # Generates a theoretical signal without noise
        x = [0 for i in range(self.N)]
        
        n = self.n0
        for i in range(self.N):
            z = complex(0, self.w0 * n * self.Ts + phase)
            x[i] = self.A * exp(z)
            n += 1

        return x
