from fast_dtft import FastDTFT
import signal_generation

from math import atan2
from cmath import exp, pi


class Estimators:
  def __init__(self, parameters):
    # Init the class with the parameters:
    #  N, M, T, n0
    self.N = parameters.get('N')
    self.M = parameters.get('M')
    self.T = parameters.get('T')
    self.n0 = parameters.get('n0')

  def calculate_m_star(self, magnitude_signal):
    # Finds the index with the maximum magnitude
    idx = 0
    max_mag = 0
    
    for i in range(len(magnitude_signal)):
      if abs(magnitude_signal[i]) > max_mag:
        idx = i

    return idx

  def F_omega0(self, signal, omega0):
    # Calculates the fourier, and normalizes it
    complex_sum = complex(0, 0)

    for n in range(len(signal)):
      complex_sum += (signal[n] * exp(complex(0, -omega0 * n * self.T)))
    
    return complex_sum / self.N

  def estimate_omega(self, signal):
    # Estimates the signals angular frequency
    F_DTFT = FastDTFT()

    x_zp = F_DTFT.zero_pad(signal, self.M)
    x_mag = F_DTFT.magnitude(x_zp)

    m_star = self.calculate_m_star(x_mag)

    return 2 * pi * m_star / (self.M * self.T)

  def estimate_phase(self, signal):
    # Estimates the phase of the signal
    omega_estimate = self.estimate_omega(signal)

    F_omega_estimate = self.F_omega0(signal, omega_estimate)

    adjusted_angle = exp(complex(0, -omega_estimate * self.n0 * self.T)) * F_omega_estimate
    return atan2(adjusted_angle.imag, adjusted_angle.real)

if __name__ == '__main__':
  parameters = {'N': 513, 'M': 1024, 'T': 1e-6, 'n0': -256}

  Estimates = Estimators(parameters)
  X = signal_generation.SampledSignal()

  signal = X.generate_sampled_signal()

  omega_estimate = Estimates.estimate_omega(signal)
  phase_estimate = Estimates.estimate_phase(signal)

  print(omega_estimate)
  print(phase_estimate)
