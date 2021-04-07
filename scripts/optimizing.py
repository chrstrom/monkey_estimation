#!/usr/bin/env python

import cfg
import statistics
import numpy as np
import matplotlib.pyplot as plt
import Signals
import estimation
import error_calculation

from scipy import optimize, fft, fftpack

"""
In this file, we should try to optimize the performance of the estimate
of omega and the phase. This is done by minimizing MSE with the help of 
the Nelder-Mead-algorithm
"""

class Optimize:
  def __init__(self, f0=None, phi0=None, M=None, SNR=None):
    self.N = cfg.N
    self.T = cfg.Ts
    self.n0 = cfg.n0
    self.Fs = cfg.Fs

    if f0 is None:
      self.f0 = cfg.f0
    else:
      self.f0 = f0

    if phi0 is None:
      self.phi0 = cfg.phi
    else:
      self.phi0 = phi0

    if M is None or M <= 0:
      self.M = 2**10
    else:
      self.M = M

    if SNR is None:
      self.SNR = cfg.SNR
    else:
      self.SNR = SNR

  def frequency_objective_function(self, x):
    # Estimate number k for angular frequency from NM-algorithm 
    w_k = x[0]

    # Creating objects 
    sig = Signals.Signals(self.SNR)
    fft_est = estimation.FFTEstimator(self.M)
    error_cal = error_calculation.ErrorCalculation()

    # Creating signals
    x_d = sig.x_discrete()
    x_w = sig.x_ang_frequency(w_k)

    Fx_d, _ = fft_est.fast_dtft(x_d)
    Fx_w, _ = fft_est.fast_dtft(x_w)

    # Tries minimizing the error with MSE
    return error_cal.mse(Fx_d, Fx_w)


  def phase_objective_function(self, x):
    # Estimate number k for the phase
    phi_k = x[0]

    # Creating objects 
    sig = Signals.Signals(self.SNR)
    error_cal = error_calculation.ErrorCalculation()

    # Creating signals
    x_d = sig.x_discrete()
    x_p = sig.x_phase(phi_k)

    # Tries minimizing the error with MSE
    return error_cal.mse(x_d, x_p)


  def optimize_frequency_nelder_mead(self, x0, max_iterations):
    self.frequencies = []
    self.mse = []

    for i in range(max_iterations):
        frequency = optimize.minimize(self.frequency_objective_function, x0, method="Nelder-Mead")
        self.frequencies.append(frequency.x[0])
        self.mse.append((self.f0 - frequency.x[0])**2)
    
    return self.frequencies, self.mse


  def optimize_phase_nelder_mead(self, x0, max_iterations):
    self.phases = []
    self.mse = []

    for i in range(max_iterations):
        phase = optimize.minimize(self.phase_objective_function, x0, method="Nelder-Mead")
        self.phases.append(phase.x[0])
        self.mse.append((self.phi0 - phase.x[0])**2)
    
    return self.frequencies, self.mse

  # Doesn't quite work
  # def plot_mse(self, min_frequency, max_frequency, frequency_step):
  #   mse = []    
  #   it = [1,2]
  #   for f in range(min_frequency, max_frequency, frequency_step):
  #       it[0] = f
  #       mse.append(self.objective_function(it))

  #   plt.figure(2)
  #   plt.title("MSE")
  #   plt.xlabel("Frequency [Hz]")
  #   plt.ylabel("Mean Square Error")
  #   plt.plot(np.arange(min_frequency, max_frequency, frequency_step), mse)


if __name__ == '__main__':
  opt = Optimize()

  ## Optimize the frequency and phase ##
  f0 = cfg.f0
  phi0 = cfg.phi
  max_iterations = 100

  # Get some warnings that I try to cast complex to real, which discrads imaginary value
  frequencies, mse_freq = opt.optimize_frequency_nelder_mead(f0, max_iterations)
  phases, mse_phase = opt.optimize_phase_nelder_mead(phi0, max_iterations)


  mean_frequency = statistics.mean(frequencies)
  mean_mse_freq = statistics.mean(mse_freq) # Somehow this is negative...
  mean_phase = statistics.mean(phases)
  mean_mse_phase = statistics.mean(mse_phase)
  
  mse_freq_variance = statistics.variance(mse_freq, mean_mse_freq)
  mse_phase_variance = statistics.variance(mse_phase, mean_mse_phase)

  print("Last optimized frequency:", frequencies[-1])
  print("Average optimized frequency:", mean_frequency)
  print("Average optimized mse:", mean_mse_freq)
  print("Average optimized mse variance:", mse_freq_variance)

  print("Last optimized phase:", phases[-1])
  print("Average optimized phase:", mean_phase)
  print("Average optimized mse:", mean_mse_phase)
  print("Average optimized mse variance:", mse_phase_variance)
