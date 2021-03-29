import numpy as np

from math import atan2
from scipy.fft import fft, ifft

class FastDTFT:
  def __init__(self):
    pass

  def zero_pad(self, signal, M):
    # zero-padding such that the system can take M-point FFT
    for i in range(M - len(signal)):
      signal.append(0)
    return signal
    # return signal.extend([0 for i in range(0, M - len(signal))])

  def fast_dtft(self, signal, M):
    # M number of fft-points
    if len(signal) < M:
      signal = zero_pad(signal)      

    return fft(signal, M)

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

    


