#!/usr/bin/env python

from fast_dtft import FastDTFT
import Signals
import cfg

from math import atan2
from cmath import exp, pi

import matplotlib.pyplot as plt
import numpy as np

class Estimators:
  def __init__(self):
    # Init the class with parameters from config-file
    self.N = cfg.N
    self.M = cfg.M
    self.T = cfg.Ts
    self.n0 = cfg.n0

  def calculate_m_star(self, magnitude_signal):
    # Finds the index with the maximum magnitude
    return np.argmax(magnitude_signal) / 6.0

  def F_omega0(self, signal, omega0):
    # Calculates the fourier and normalizes it
    complex_sum = complex(0, 0)

    for n in range(0, len(signal)):
      complex_sum += (signal[n] * exp(complex(0, -omega0 * n * self.T)))
    
    return complex_sum / self.N

  def estimate_omega(self, signal):
    # Estimates the signals angular frequency
    F_DTFT = FastDTFT()

    x_f, Ff = F_DTFT.fast_dtft(signal)
    x_mag = F_DTFT.magnitude(x_f)
    m_star = self.calculate_m_star(x_mag)

    return 2 * pi * m_star / (self.M * self.T)

  def estimate_phase(self, signal):
    # Estimates the phase of the signal
    omega_estimate = self.estimate_omega(signal)

    F_omega_estimate = self.F_omega0(signal, omega_estimate)

    adjusted_angle = exp(complex(0, -omega_estimate * self.n0 * self.T)) * F_omega_estimate
    return atan2(adjusted_angle.imag, adjusted_angle.real)

if __name__ == '__main__':
  Estimates = Estimators()
  SG = Signals.Signals()

  signal = SG.x_discrete()

  omega_estimate = Estimates.estimate_omega(signal)
  phase_estimate = Estimates.estimate_phase(signal)

  if phase_estimate < 0:
    phase_estimate += pi

  print(omega_estimate)
  print(phase_estimate)
