#!/usr/bin/env python

import signals
import cfg

import numpy as np

def calculate_m_star(Fw):
    """
    Calculate the m_star as described by 
    Eq. (5) and Eq. (9) in the project spec.
    """
    return np.argmax(np.absolute(Fw))

def calculate_w_hat(m_star, M):
    """
    Calculate the angular frequency estimate
    w_FFT_hat as described by Eq. (8) in the
    project spec
    """
    return 2*np.pi*m_star / (M*cfg.Ts)

def calculate_phi_hat(x_d, w_hat):
    """
    Calculate the phase estimate phi_hat as
    described by Eq. (7) in the project spec
    """
    F_w_hat = signals.F(x_d, w_hat)
    phi_arg = np.exp(-1j*w_hat*cfg.n0*cfg.Ts)*F_w_hat
    return np.angle(phi_arg)

def estimator(x_d, M):
    """
    Use the M-point FFT estimator described in the
    problem spec. to estimate the angular frequency
    "w" and phase "phi" of the input signal x_d.

    Returns the estimates w_hat and phi_hat, as well
    as the FFT of the input signal, for data analysis
    purposes.
    """
    Fw = np.fft.fft(x_d, M)
    m_star = calculate_m_star(Fw)

    w_hat = calculate_w_hat(m_star, M)
    phi_hat = calculate_phi_hat(x_d, w_hat)

    return w_hat, phi_hat, Fw