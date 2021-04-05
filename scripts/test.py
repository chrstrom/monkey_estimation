#!/usr/bin/env python

import fast_dtft 
from Signals import Signals

import matplotlib.pyplot as plt

if __name__ == '__main__':
  sig = Signals()
  x = sig.x_discrete()

  k = 20
  M = pow(2, k)
  print(M)

  f_dtft = fast_dtft.FastDTFT()
  # x_zp = f_dtft.zero_pad(x, M)

  Fx = f_dtft.fast_dtft(x)
  
  # print(x)
  # print(X)

  # x_abs = F_DTFT.magnitude(x)
  # X_abs = F_DTFT.magnitude(X)
  # X_pha = F_DTFT.phase(X)

  # plt.plot(X_abs)
  # plt.show()

  # plt.plot(X_pha)
  # plt.show()
