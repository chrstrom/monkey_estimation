#!/usr/bin/env python

import cfg
import numpy as np

def sigma_squared_from_SNR(SNR):
    """
    Calculate the value for sigma^2 according to the
    definition.
    """
    return cfg.A**2 / (2*float(SNR)) # Float casting prevents floor division

def F(x_d, w):
    """
    Calculate F(w) according to Eq. (6) in the project
    specifications.
    """
    sum = 0
    for n in range(0, cfg.N):
        sum += x_d[n]*np.exp(-1j*w*n*cfg.Ts)

    return sum / cfg.N

def generate_signal(sigma):
    """
    Generate a signal according to the problem spec.
    which consists of a complex exponential with 
    added noise. Note that the input argument is sigma,
    NOT sigma^2
    """
    wr = np.random.normal(0, sigma, cfg.N)
    wi = np.random.normal(0, sigma, cfg.N)
    w = wr + 1j*wi

    x = np.empty(cfg.N, dtype=np.complex_)
    for n in range(cfg.N):
        x[n] = cfg.A*np.exp(1j*(cfg.w0*(n+cfg.n0)*cfg.Ts + cfg.phi))

    return x + w