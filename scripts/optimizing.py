#!/usr/bin/env python

import cfg
import numpy as np
import matplotlib.pyplot as plt

import signals
import fft_estimator

from datetime import datetime as dt
from scipy import optimize

"""
In this file, we should try to optimize the performance of the estimate
of omega and the phase. This is done by minimizing MSE with the help of 
the Nelder-Mead-algorithm
"""

class Optimize:
  def __init__(self, SNR, f0=None, phi0=None, M=None):
    self.N = cfg.N
    self.T = cfg.Ts
    self.n0 = cfg.n0
    self.Fs = cfg.Fs
  
    self.SNR = SNR

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


  def mse(self, list_lhs, list_rhs):
    assert(len(list_lhs) == len(list_rhs))
    return np.square(np.absolute(list_lhs - list_rhs)).mean()
    

  def frequency_objective_function(self, x):
    # Estimate number k for frequency from NM-algorithm 
    f_k = x[0]

    # Creating signals
    x_d = signals.generate_signal(self.SNR)
    x_f = signals.x_frequency(f_k)

    Fx_d, _ = fft_estimator.M_point_fft(x_d, self.M)
    Fx_f, _ = fft_estimator.M_point_fft(x_f, self.M)

    # Tries minimizing the error with MSE
    return self.mse(np.absolute(Fx_d), np.absolute(Fx_f))


  def phase_objective_function(self, x):
    # Estimate number k for the phase
    phi_k = x[0]

    # Creating signals
    x_d = signals.generate_signal(self.SNR)
    x_p = signals.x_phase(phi_k)

    # Tries minimizing the error with MSE
    return self.mse(x_d, x_p)


  def optimize_frequency_nelder_mead(self, x0, max_iterations):
    frequencies = np.zeros(max_iterations)
    mse = np.zeros(max_iterations)

    for i in range(max_iterations):
        frequency = optimize.minimize(self.frequency_objective_function, x0, method="Nelder-Mead")
        frequencies[i] = frequency.x[0]
        mse[i] = (self.f0 - frequency.x[0])**2
    
    return frequencies, mse


  def optimize_phase_nelder_mead(self, x0, max_iterations):
    phases = np.zeros(max_iterations)
    mse = np.zeros(max_iterations)

    for i in range(max_iterations):
        phase = optimize.minimize(self.phase_objective_function, x0, method="Nelder-Mead")
        phases[i] = phase.x[0]
        mse[i] = (self.phi0 - phase.x[0])**2
    
    return phases, mse

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
  opt = Optimize(10)

  ## Optimize the frequency and phase ##
  f0 = 1.5e5
  phi0 = np.pi / 2.0
  max_iterations = 10

  begin = dt.now()
  frequencies, mse_freq = opt.optimize_frequency_nelder_mead(f0, max_iterations)
  phases, mse_phase = opt.optimize_phase_nelder_mead(phi0, max_iterations)


  mean_frequency = np.mean(frequencies)
  # mean_mse_freq = np.mean(mse_freq) 
  mean_phase = np.mean(phases)
  # mean_mse_phase = np.mean(mse_phase)
  
  # mse_freq_variance = np.variance(mse_freq, mean_mse_freq)
  # mse_phase_variance = np.variance(mse_phase, mean_mse_phase)
  end = dt.now()
  print("Calculation time: %f seconds" % float((end - begin).total_seconds()))

  print("Last optimized frequency:", frequencies[-1])
  print("Average optimized frequency:", mean_frequency)
  # print("Average optimized mse:", mean_mse_freq)
  # print("Average optimized mse variance:", mse_freq_variance)

  print("Last optimized phase:", phases[-1])
  print("Average optimized phase:", mean_phase)
  # print("Average optimized mse:", mean_mse_phase)
  # print("Average optimized mse variance:", mse_phase_variance)
