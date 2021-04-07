#!/usr/bin/env python

# A single file containing everything needed for the estimation
import numpy as np
import matplotlib.pyplot as plt

def generate_crlb(sigma_squared):
    crlb_w   = 12*sigma_squared / (A**2 * T**2 * N*(N**2 - 1))
    crlb_phi = 12*sigma_squared*(n0**2 * N + 2*n0*P + Q) / (A**2 * N**2 * (N**2 - 1))

    return crlb_w, crlb_phi

def generate_signal(sigma_squared):
    wr = np.random.normal(0, sigma_squared, N)
    wi = np.random.normal(0, sigma_squared, N)
    w = wr + 1j*wi

    x = np.empty(N, dtype=np.complex_)
    for n in range(N):
        x[n] = A*np.exp(1j*(w0*(n+n0)*T + phi)) + w[n]

    return x

def fft_estimator(x_d, M):
    Fx = np.fft.fft(x_d, M)

    m_star = np.argmax(np.absolute(Fx))
    w_hat = 2*np.pi*m_star / (M*T)

    F_w_hat = 0
    for n in range(N):
        F_w_hat += x_d[n]*np.exp(-1j*w_hat*n*T)
    F_w_hat /= N

    phi_arg = np.exp(-1j*w_hat*n0*T)*F_w_hat
    phi_hat = np.angle(phi_arg)

    return w_hat, phi_hat


# Constants
Fs = 10**6
T = 1.0/Fs

f0 = 10**5
w0 = 2*np.pi*f0

phi = np.pi / 8

A = 1.0
N = 513

P = N*(N-1)/2.0
Q = N*(N-1)*(2*N-1)/6.0

n0 = -P/N

# Generate multiple samples to calculate variance
SNR_dB = 0.0
SNR = 10**(SNR_dB/10.0)

K = 10
M = 2**K

sigma_squared = A**2 / (2*SNR)

NUM = 100

w_estimates = np.empty(NUM)
phi_estimates = np.empty(NUM)

for i in range(NUM):

    x_d = generate_signal(sigma_squared)
    w_hat, phi_hat = fft_estimator(x_d, M)

    w_estimates[i] = w_hat
    phi_estimates[i] = phi_hat
    
var_w = np.var(w_estimates)
var_phi = np.var(phi_estimates)
crlb_w, crlb_phi = generate_crlb(sigma_squared)

print("SNR [dB]: {}, M: 2^{}".format(SNR_dB, K))
print("OMEGA | estimated variance: {}, crlb: {}".format(var_w, crlb_w))
print("PHASE | estimated variance: {}, crlb: {}".format(var_phi, crlb_phi))