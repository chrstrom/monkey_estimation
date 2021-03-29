#!/usr/bin/env python

import numpy as np
import cfg 

from math import atan2
from scipy import fft, ifft


class FastDTFT:
  def __init__(self):
    self.M = cfg.M

  def zero_pad(self, signal):
    # Must zero-pad such that the system can take M-point FFT
    return signal.extend([0 for i in range(0, self.M - len(signal))])

  def fast_dtft(self, signal):
    # M number of fft-points
    assert len(signal) == self.M      

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

    