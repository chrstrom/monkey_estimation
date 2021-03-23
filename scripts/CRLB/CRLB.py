#!/usr/bin/env python
import sys
sys.path.append("..")
from config import cfg

class CRLB:

    N = cfg.N
    T = cfg.Ts
    A = cfg.A
    n0 = cfg.n0

    P = cfg.P
    Q = cfg.Q

    def omega(self, SNR):

        sigma_squared = self.sigma_squared_from_SNR(SNR)

        numerator = 12*sigma_squared
        denominator = self.A**2 * self.T**2 * self.N*(self.N**2 - 1)

        return numerator / denominator

    def phi(self, SNR):
        sigma_squared = self.sigma_squared_from_SNR(SNR)

        numerator = 12*sigma_squared*(self.n0**2 * self.N + 2*self.n0*self.P + self.Q)
        denominator = self.A**2 * self.N**2 * (self.N**2 - 1)

        return numerator / denominator

    def sigma_squared_from_SNR(self, SNR):
        return self.A**2 / (2*float(SNR))



if __name__ == '__main__':
    crlb = CRLB()

    print(crlb.omega(1))
