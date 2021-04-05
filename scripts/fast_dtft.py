#!/usr/bin/env python

import numpy as np
import cfg 

from math import atan2
from scipy import fft, ifft, fftpack


class FastDTFT:
  def __init__(self, M=None):
    
    if M is None:
      self.M = cfg.M
    else:
      self.M = M

    self.Fs = cfg.Fs

  def zero_pad(self, signal):
    # zero-padding such that the system can take M-point FFT
    for i in range(self.M - len(signal)):
      signal.append(0)
    return signal

  def fast_dtft(self, signal):
    # M number of fft-points
    if len(signal) < self.M:
      signal = self.zero_pad(signal)       

    Fx = fft(signal, self.M)
    Ff = fftpack.fftfreq(self.M, 1 / self.Fs)

    Fx = fftpack.fftshift(Fx)
    Ff = fftpack.fftshift(Ff)
    
    return Fx, Ff

  def magnitude(self, signal):
    # Return || signal ||_{2} ^ 2
    magnitude = [0 for i in range(len(signal))]
    
    for i in range(len(signal)):
      magnitude[i] = abs(signal[i])

    return magnitude

  def phase(self, signal):
    # Calculates the phase of a given signal
    # Each time-point must later be merged with the equivalent frequency to
    # get the optimal response
    phase = [0 for i in range(len(signal))] 

    for i in range(len(signal)):
      try:
        phase[i] = atan2(signal[i].imag, signal[i].real)
      except Exception as e:
        print((signal[i].imag, signal[i].real))
        phase[i] = 0    
    return phase

    