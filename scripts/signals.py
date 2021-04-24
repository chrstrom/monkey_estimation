#!/usr/bin/env python

import sys
import os

if os.name == 'nt':
    sys.path[0]=os.path.dirname(os.path.realpath(__file__))
    
import cfg
import numpy as np

def sigma_squared_from_SNR_dB(SNR_dB):
    """
    Calculate the value for sigma^2 according to the
    definition.
    """
    SNR = 10**(SNR_dB/10.0)
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

def x_discrete(SNR_dB):
    """
    Generate a signal according to the problem spec.
    which consists of a complex exponential with 
    added noise. Noise-to-signal ratio defined by SNR
    """
    sigma = np.sqrt(sigma_squared_from_SNR_dB(SNR_dB))

    wr = np.random.normal(0, sigma, cfg.N)
    wi = np.random.normal(0, sigma, cfg.N)
    w = wr + 1j*wi

    x = np.empty(cfg.N, dtype=np.complex_)

    for n in range(cfg.N):
        z = 1j*(cfg.w0 * (n+cfg.n0) * cfg.Ts + cfg.phi)
        x[n] = cfg.A * np.exp(z)

    return x + w


def x_ideal(frequency, phase):
    """ 
    Generates a complex-exponential signal with given frequency
    and phase. Does not contain noise
    """
    x = np.empty(cfg.N, dtype=np.complex_)
    
    for n in range(cfg.N):
        z = 1j*(2 * np.pi * frequency * (cfg.n0+n) * cfg.Ts + phase)
        x[n] = cfg.A * np.exp(z)

    return x