from fft import FFT
from signal_generation import SampledSignal

import matplotlib.pyplot as plt

if __name__ == '__main__':
  SG = SampledSignal()
  x = SG.calculate_signal()

  M = 100000

  F_DFT = FFT()
  x_zp = F_DFT.zero_pad(x, M)

  X = F_DFT.fast_dft(x, M)
  
  # print(x)
  # print(X)

  # x_abs = F_DFT.magnitude(x)
  X_abs = F_DFT.magnitude(X)
  X_pha = F_DFT.phase(X)

  plt.plot(X_abs)
  plt.show()

  plt.plot(X_pha)
  plt.show()
