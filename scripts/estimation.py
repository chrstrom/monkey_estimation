#!/usr/bin/env python

from fast_dtft import FastDTFT
import Signals
import cfg

from math import atan2
from cmath import exp, pi

import numpy as np

class Estimators:
  def __init__(self):
    # Init the class with parameters from config-file
    self.N = cfg.N
    self.M = cfg.M
    self.T = cfg.Ts
    self.n0 = cfg.n0

  def calculate_m_star(self, signal):
    # Finds the index with the maximum magnitude
    m = np.argmax(signal)
    return m

  def F_omega0(self, signal, omega0):
    # Calculates the fourier and normalizes it
    complex_sum = complex(0, 0)

    for n in range(0, len(signal)):
      complex_sum += (signal[n] * exp(complex(0, -omega0 * n * self.T)))
      #print(complex_sum)
    
    return complex_sum / self.N

  def estimate_omega(self, signal):
    # Estimates the signals angular frequency
    F_DTFT = FastDTFT()

    x_zp = F_DTFT.zero_pad(signal, self.M)
    x_fft, x_freq = F_DTFT.fast_dtft(x_zp)

    m_star = self.calculate_m_star(x_fft)

    omega_hat = 2 * pi * m_star / (self.M * self.T)
    return omega_hat

  def estimate_phase(self, signal):
    # Estimates the phase of the signal
    omega_estimate = self.estimate_omega(signal)

    F_omega_estimate = self.F_omega0(signal, omega_estimate)

    adjusted_angle = exp(complex(0, -omega_estimate * self.n0 * self.T)) * F_omega_estimate
    phi_hat = atan2(adjusted_angle.imag, adjusted_angle.real)

    if phi_hat < 0:
      phi_hat += pi
      
    return phi_hat

if __name__ == '__main__':
  Estimates = Estimators()
  SG = Signals.Signals()

  signal = SG.x_discrete()

  for i in range(len(signal)):
    if signal[i] == complex(0,0):
      print("May be zero-padding error at index ", i)

  omega_estimate = Estimates.estimate_omega(signal)
  phase_estimate = Estimates.estimate_phase(signal)

  print("True omega: %s, calculated omega")
  print(omega_estimate/cfg.Fs)
  print(phase_estimate)
