#!/usr/bin/env python

import cfg
import signals as sig


def omega(SNR):

    sigma_squared = sig.sigma_squared_from_SNR(SNR)

    numerator = 12*sigma_squared
    denominator = cfg.A**2 * cfg.Ts**2 * cfg.N*(cfg.N**2 - 1)

    return numerator / denominator

def phi(SNR):
    sigma_squared = sig.sigma_squared_from_SNR(SNR)

    numerator = 12*sigma_squared*(cfg.n0**2 * cfg.N + 2*cfg.n0*cfg.P + cfg.Q)
    denominator = cfg.A**2 * cfg.N**2 * (cfg.N**2 - 1)

    return numerator / denominator
