#!/usr/bin/env python

import Signals
import cfg
from scipy import fft, fftpack

from math import atan2
from cmath import exp, pi

import numpy as np

class FFTEstimator:
  def __init__(self, M):
    # Init the class with parameters from config-file
    self.N = cfg.N
    self.T = cfg.Ts
    self.n0 = cfg.n0
    self.Fs = cfg.Fs

    self.M = M


  def fast_dtft(self, signal):    
    Fx = fft(signal, self.M)
    Ff = fftpack.fftfreq(self.M, 1 / self.Fs)

    Fx = fftpack.fftshift(Fx)
    Ff = fftpack.fftshift(Ff)
    
    return Fx, Ff


  def magnitude(self, signal):
    # Return || signal ||_{2} ^ 2
    magnitude = np.zeros(len(signal))
    
    for i in range(len(signal)):
      magnitude[i] = abs(signal[i])

    return magnitude


  def calculate_m_star(self, signal):
    # Finds the index with the maximum magnitude
    m = np.argmax(signal)

    # Change m to counteract the shift in FFT
    if m > self.M/2:
      m = m - self.M/2
    else:
      m = self.M/2 - m

    return m


  def F_omega0(self, x_d, omega0):
    # Calculates the fourier and normalizes it
    complex_sum = complex(0, 0)

    for n in range(0, len(x_d)):
      complex_sum += (x_d[n] * exp(complex(0, -omega0 * n * self.T)))
    
    return complex_sum / self.N


  def estimate_omega(self, x_d):
    # Estimates the signals angular frequency
    x_f, _ = self.fast_dtft(x_d)
    x_mag = self.magnitude(x_f)

    m_star = self.calculate_m_star(x_mag)

    omega_hat = 2 * pi * m_star / (self.M * self.T)
    return omega_hat


  def estimate_phase(self, x_d):
    # Estimates the phase of the signal
    omega_estimate = self.estimate_omega(x_d)

    F_omega_estimate = self.F_omega0(x_d, omega_estimate)

    adjusted_angle = exp(complex(0, -omega_estimate * self.n0 * self.T)) * F_omega_estimate
    phi_hat = atan2(adjusted_angle.imag, adjusted_angle.real)

    if phi_hat < 0:
      phi_hat += pi
      
    return phi_hat
