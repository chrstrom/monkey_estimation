#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from scripts import Signals
from scripts import fast_dtft
from scripts import cfg

import scipy

sig = Signals.Signals()
FFT = fast_dtft.FastDTFT()

x = sig.x_discrete()
Fx, Ff = FFT.fast_dtft(x)

plt.plot(Ff, abs(Fx))
plt.xlabel("Frequency [Hz]")
plt.ylabel("abs(Fx)")
plt.title("%d-point Fourier transform of x[n]" % cfg.M)
plt.show()
