#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scripts import Signals
from scripts import cfg

from scipy import fft, ifft, fftpack

def fast_dtft(self, signal):  
  Fx = fft.fft(signal, self.M)
  Ff = fftpack.fftfreq(self.M, 1 / self.Fs)

  Fx = fftpack.fftshift(Fx)
  Ff = fftpack.fftshift(Ff)
  
  return Fx, Ff


if __name__ == 'main':
  sig = Signals.Signals()
  fft = fast_dtft.FastDTFT()

  x = sig.x_discrete()
  Fx, Ff = fft.fast_dtft(x)

  plt.plot(Ff, abs(Fx))
  plt.xlabel("Frequency [Hz]")
  plt.ylabel("abs(Fx)")
  plt.title("%d-point Fourier transform of x[n]" % cfg.M)
  plt.show()
