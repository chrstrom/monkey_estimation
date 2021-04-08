#!/usr/bin/env python

# A single file containing everything needed for the estimation
import numpy as np
import matplotlib.pyplot as plt

def generate_crlb(sigma_squared):
    crlb_w   = 12*sigma_squared / (A**2 * T**2 * N*(N**2 - 1))
    crlb_phi = 12*sigma_squared*(n0**2 * N + 2*n0*P + Q) / (A**2 * N**2 * (N**2 - 1))

    return crlb_w, crlb_phi

def generate_signal(sigma):
    # Noise generation
    wr = np.random.normal(0, sigma, N)
    wi = np.random.normal(0, sigma, N)
    w = wr + 1j*wi

    # Pure signal generation
    x = np.empty(N, dtype=np.complex_)
    for n in range(N):
        x[n] = A*np.exp(1j*(w0*(n+n0)*T + phi))

    return x + w

def F(x_d, w):
    Fw0 = 0
    for n in range(N):
        Fw0 += x_d[n]*np.exp(-1j*w*n*T) # Eq. (6)
    Fw0 /= N

    return Fw0

def fft_estimator(x_d, M):
    Fw = np.fft.fft(x_d, M)
    m_star = np.argmax(np.absolute(Fw))
    
    w_hat = 2*np.pi*m_star / (M*T)   # Eq. (8)
    phi_hat = np.angle(np.exp(-1j*w_hat*n0*T)*F(x_d, w_hat))   # Eq. (7)

    return w_hat, phi_hat, Fw

def plot(x_d, Fw, i):
    Fw = np.fft.fftshift(Fw)
    Fw = np.absolute(Fw)

    Ff = np.fft.fftfreq(M, 1.0/Fs)
    Ff = np.fft.fftshift(Ff)

    if i == 0:
        plt.figure(1)
        plt.plot(np.arange(len(x_d)), np.real(x_d))
        plt.title("First generated signal with GWN, real values")

        plt.figure(2)
        plt.plot(np.arange(len(x_d)), np.imag(x_d))
        plt.title("First generated signal with GWN, imag values")

        plt.figure(3)
        plt.plot(Ff, Fw)
        plt.title("Abs. of F-transform of first generated signal")

    plt.figure(4)
    plt.plot(Ff, Fw)
    plt.title("F-transforms of all generated signals")

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
SNR_dB = 10.0
SNR = 10**(SNR_dB/10.0)

K = 10
M = 2**K

sigma_squared = A**2 / (2*SNR)

NUM = 100
w_estimates = np.empty(NUM)
phi_estimates = np.empty(NUM)
do_plot = True

for i in range(NUM):

    x_d = generate_signal(np.sqrt(sigma_squared))
    w_hat, phi_hat, Fw = fft_estimator(x_d, M)

    w_estimates[i] = w_hat
    phi_estimates[i] = phi_hat

    if do_plot:
        plot(x_d, Fw, i)
    
plt.show()

mean_w = np.mean(w_estimates)
mean_phi = np.mean(phi_estimates)

var_w = np.var(w_estimates)
var_phi = np.var(phi_estimates)

crlb_w, crlb_phi = generate_crlb(sigma_squared)

print("CONFIG | SNR [dB]: {}, M: 2^{}, true omega: {}, true phase: {}".format(SNR_dB, K, w0, phi))
print("OMEGA  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_w, var_w, crlb_w))
print("PHASE  | estimated mean: {}, estimated variance: {}, crlb: {}".format(mean_phi, var_phi, crlb_phi))