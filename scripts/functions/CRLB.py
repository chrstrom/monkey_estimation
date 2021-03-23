#!/usr/bin/env python

class CRLB:

    N = 513
    T = 10e-6
    A = 1
    n0 = -256

    P = N * (N-1) / 2
    Q = N * (N-1) * (2*N-1) / 6

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
