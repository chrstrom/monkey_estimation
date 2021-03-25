#!/usr/bin/env python

import cfg
import Signals as sig

class CRLB:

    N = cfg.N
    T = cfg.Ts
    A = cfg.A
    n0 = cfg.n0

    P = cfg.P
    Q = cfg.Q
    
    SNR = cfg.SNR

    def omega(self):

        sigma_squared = sig.sigma_squared_from_SNR(self.A, self.SNR)

        numerator = 12*sigma_squared
        denominator = self.A**2 * self.T**2 * self.N*(self.N**2 - 1)

        return numerator / denominator

    def phi(self):
        sigma_squared = sig.sigma_squared_from_SNR(self.A, self.SNR)

        numerator = 12*sigma_squared*(self.n0**2 * self.N + 2*self.n0*self.P + self.Q)
        denominator = self.A**2 * self.N**2 * (self.N**2 - 1)

        return numerator / denominator



if __name__ == '__main__':
    crlb = CRLB()

    print(crlb.omega())
