#!/usr/bin/env python

from cmath import pi

k = 22

N = 513
M = 2**k

Fs = 1e6
Ts = 1.0 / Fs

A = 1
phi = pi / 8.0

SNR_dB = 90.0              # [-10, 60]
SNR = 10**(SNR_dB/10.0)

f0 = 1e5
w0 = 2 * pi * f0 

P = N * (N - 1) / 2.0
Q = N * (N - 1) * (2 * N - 1) / 6.0

n0 = int(-P / N)
