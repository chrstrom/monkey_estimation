import fast_dtft 
from signal_generation import SampledSignal
from Signals import Signals

import matplotlib.pyplot as plt

if __name__ == '__main__':
  SG = Signals()
  x = SG.x_discrete()

  k = 20
  M = pow(2, k)
  print(M)

  F_DTFT = fast_dtft.FastDTFT()
  x_zp = F_DTFT.zero_pad(x, M)

  X = F_DTFT.fast_dtft(x)
  
  # print(x)
  # print(X)

  # x_abs = F_DTFT.magnitude(x)
  X_abs = F_DTFT.magnitude(X)
  X_pha = F_DTFT.phase(X)

  plt.plot(X_abs)
  plt.show()

  plt.plot(X_pha)
  plt.show()
