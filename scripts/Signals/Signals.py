from cmath import exp, pi
import numpy as np

class Signals:

    N = 513
    T = 10e-6
    A = 1
    Fs = 1e6
    Ts = 1.0 / Fs
    phi = pi / 8.0
    sigma = 1.0

    f0 = 1e5
    w0 = 2 * pi * f0 

    P = N * (N - 1) / 2.0
    Q = N * (N - 1) * (2 * N - 1) / 6.0

    n0 = -P / N

    def F(self, w0):
        x = self.x_discrete()
        sum = 0
        for n in range(0, self.N):
            sum += x[n]*exp(-1j*w0*n*self.T)

        return sum / self.N

    def x_discrete(self):
        # Generate the data for the sampled signal
        x = [0 for i in range(self.N)]
        noise = np.random.normal(0, pow(self.sigma, 2), self.N)

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
