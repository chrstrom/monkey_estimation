#!/usr/bin/env python

from cmath import exp, pi
import numpy as np
import matplotlib.pyplot as plt

import cfg
import fast_dtft


def sigma_squared_from_SNR(A, SNR):
    return A**2 / (2*float(SNR))

class Signals:

    N   = cfg.N
    Fs  = cfg.Fs
    Ts  = cfg.Ts
    A   = cfg.A

    P   = cfg.P
    Q   = cfg.Q
    n0  = cfg.n0

    phi = cfg.phi

    f0  = cfg.f0
    w0  = cfg.w0

    SNR = cfg.SNR
    sigma = sigma_squared_from_SNR(A, SNR)

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

if __name__ == '__main__':
    obj = fast_dtft.FastDTFT()
    sig = Signals()
    print(sig.F(0.1))

    x = np.array(sig.x_discrete())

    # signal = obj.zero_pad(x)
    fourier, frequencies = obj.fast_dtft(x)

    norm = np.linalg.norm(fourier)
    plt.plot(np.absolute(fourier).tolist() / norm)
    plt.show()

