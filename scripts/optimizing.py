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
  def __init__(self, f0=None, M=None, SNR=None):
    self.N = cfg.N
    self.T = cfg.Ts
    self.n0 = cfg.n0
    self.Fs = cfg.Fs

    if f0 is None:
      self.f0 = cfg.f0
    else:
      self.f0 = f0

    if M is None or M <= 0:
      self.M = 2**10
    else:
      self.M = M

    if SNR is None:
      self.SNR = cfg.SNR
    else:
      self.SNR = SNR

  def objective_function(self, x):
    # Estimate number k for angular frequency from NM-algorithm 
    w_k = x[0]

    # Creating objects 
    sig = Signals.Signals(self.SNR)
    fft_est = estimation.FFTEstimator(self.M)
    error_cal = error_calculation.ErrorCalculation()

    # Creating signals
    x_d = sig.x_discrete()
    x_c = sig.x_theoretical(w_k)

    Fx_d, _ = fft_est.fast_dtft(x_d)
    Fx_c, _ = fft_est.fast_dtft(x_c)

    # Tries minimizing the error with MSE
    return error_cal.mse(Fx_d, Fx_c)


  def optimize_mse_nelder_mead(self, x0, max_iterations):
    self.estimates = []
    self.mse = []

    for i in range(max_iterations):
        estimate = optimize.minimize(self.objective_function, x0, method="Nelder-Mead")
        self.estimates.append(estimate.x[0])
        self.mse.append(self.f0 - estimate.x[0])
    
    return self.estimates, self.mse

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

  x0 = 1e5
  max_iterations = 100

  # Get some warnings that I try to cast complex to real, which discrads imaginary value
  iterates, mse = opt.optimize_mse_nelder_mead(x0, max_iterations)
    
  mean_iterations = statistics.mean(iterates)
  mean_mse = statistics.mean(mse)
  
  mse_variance = statistics.variance(mse, mean_mse)

  print("Average frequency:", mean_iterations)
  print("Average mse:", mean_mse)
  print("Average mse variance:", mse_variance)

  # opt.plot_mse(50000, 150000, 100)
