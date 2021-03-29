#!/usr/bin/env python

import numpy as np
import cfg 

from math import atan2
from scipy import fft, ifft


class FastDTFT:
  def __init__(self):
    self.M = cfg.M

  def zero_pad(self, signal, M):
    # zero-padding such that the system can take M-point FFT
    for i in range(M - len(signal)):
      signal.append(0)
    return signal

  def fast_dtft(self, signal):
    # M number of fft-points
    if len(signal) < M:
      signal = zero_pad(signal)      

    return fft(signal, self.M)

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
      phase[i] = -atan2(signal[i].imag, signal[i].real)
    
    return phase

    