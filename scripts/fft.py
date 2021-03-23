import numpy as np

from math import atan2
from scipy.fft import fft, ifft

class FFT:
  def __init__(self):
    pass

  def zero_pad(self, signal, M):
    # Must zero-pad such that the system can take fft correctly
    return signal.extend([0 for i in range(0, M - len(signal))])

  def fast_dft(self, signal, M):
    # M number of fft-points

    return fft(signal, M)

  def magnitude(self, signal):
    # Return || signal ||_{2} ^ 2
    magnitude = [0 for i in range(len(signal))]
    
    for i in range(len(signal)):
      magnitude[i] = abs(signal[i])

    return magnitude

  def phase(self, signal):
    phase = [0 for i in range(len(signal))] 

    for i in range(len(signal)):
      phase[i] = -atan2(signal[i].imag, signal[i].real)
    
    return phase

    


