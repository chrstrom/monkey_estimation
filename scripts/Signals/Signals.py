from cmath import exp, pi
import numpy as np

import sys
sys.path.append("..")
from config import cfg

class Signals:

    N   = cfg.N
    Fs  = cfg.Fs
    Ts  = cfg.Ts
    A   = cfg.A

    P   = cfg.P
    Q   = cfg.Q
    n0  = cfg.n0

    phi = cfg.phi
    sigma_noise = cfg.sigma_noise

    f0  = cfg.f0
    w0  = cfg.w0

    def F(self, w0):
        x = self.x_discrete()
        sum = 0
        for n in range(0, self.N):
            sum += x[n]*exp(-1j*w0*n*self.Ts)

        return sum / self.N

    def x_discrete(self):
        # Generate the data for the sampled signal
        x = [0 for i in range(self.N)]
        noise = np.random.normal(0, pow(self.sigma_noise, 2), self.N)

        n = self.n0
        for i in range(self.N):
            z = complex(0, self.w0 * n * self.Ts + self.phi)
            x[i] = self.A * exp(z) + noise[i]
            n += 1

        return x

if __name__ == '__main__':
    sig = Signals()
    print(sig.F(0.1))

    x = sig.x_discrete()
    print(x[1])
    print(len(x))
