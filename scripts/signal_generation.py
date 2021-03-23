import numpy as np
from cmath import exp, pi


class SampledSignal:
  def __init__(self):
    # Get some variables loaded - import these later 
    self.A = 1
    self.Fs = 1e6
    self.Ts = 1.0 / self.Fs
    self.phi = pi / 8.0
    self.sigma = 1.0

    self.f0 = 1e5
    self.omega0 = 2 * pi * self.f0 

    self.SNR = pow(self.A, 2) / float((2 * pow(self.sigma, 2)))

    self.N = 513

    self.P = self.N * (self.N - 1) / 2.0
    self.Q = self.N * (self.N - 1) * (2 * self.N - 1) / 6.0

    self.n0 = -self.P / self.N

  def calculate_signal(self):
    # Generate the data for the sampled signal
    x = [0 for i in range(self.N)]
    noise = np.random.normal(0, pow(self.sigma, 2), self.N)

    n = self.n0
    omega0 = self.omega0
    Ts = self.Ts
    phi = self.phi

    for i in range(self.N):
      z = complex(0, omega0 * n * Ts + phi)
      x[i] = self.A * exp(z) + noise[i]
      n += 1
    
    return x


if __name__ == "__main__":
  test = SampledSignal()

  x = test.calculate_signal()
  print(x)