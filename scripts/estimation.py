#!/usr/bin/env python

import Signals
import cfg

from math import atan2
from cmath import exp, pi

import numpy as np

class Estimators:
  def __init__(self, M=None):
    # Init the class with parameters from config-file
    self.N = cfg.N
    self.T = cfg.Ts
    self.n0 = cfg.n0

    if M is None:
      self.M = cfg.M
    else:
      self.M = M

  def calculate_m_star(self, signal):
    # Finds the index with the maximum magnitude
    m = np.argmax(signal)

    # Change m to counteract the shift in FFT
    if m > self.N:
      m = m - self.N
    else:
      m = self.N - m

    return m

  def F_omega0(self, x_d, omega0):
    # Calculates the fourier and normalizes it
    complex_sum = complex(0, 0)

    for n in range(0, len(x_d)):
      complex_sum += (x_d[n] * exp(complex(0, -omega0 * n * self.T)))
    
    return complex_sum / self.N

  def estimate_omega(self, x_fft):
    m_star = self.calculate_m_star(x_fft)

    omega_hat = 2 * pi * m_star / (self.M * self.T)
    return omega_hat

  def estimate_phase(self, x_d, x_fft):
    # Estimates the phase of the signal
    omega_estimate = self.estimate_omega(x_fft)

    F_omega_estimate = self.F_omega0(x_d, omega_estimate)

    adjusted_angle = exp(complex(0, -omega_estimate * self.n0 * self.T)) * F_omega_estimate
    phi_hat = atan2(adjusted_angle.imag, adjusted_angle.real)

    if phi_hat < 0:
      phi_hat += pi
      
    return phi_hat

#if __name__ == '__main__':
#  est = Estimators()
#  sig = Signals.Signals()
#
#  signal = sig.x_discrete()
#
#  for i in range(len(signal)):
#    if signal[i] == complex(0,0):
#      print("May be zero-padding error at index ", i)
#
#  omega_estimate = est.estimate_omega(signal)
#  phase_estimate = est.estimate_phase(signal)
#
#  print("True omega: {}, estimated omega: {}".format(cfg.f0*2*pi, omega_estimate))
#  print("True phase: {}, estimated phase: {}".format(cfg.phi, phase_estimate))
