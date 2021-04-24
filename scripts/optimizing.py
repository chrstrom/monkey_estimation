#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
from collections import Counter

from scipy import optimize
from math import pi, floor
import numpy as np

import fft_estimator
import signals
import cfg


class Optimize:
  def __init__(self, f0=None, phi0=None, M=None, SNR=None, num_optimizations=None):
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

    if num_optimizations is None:
      self.num_opt = cfg.num_opt
    else:
      self.num_opt = num_optimizations
  
  def mse(self, list_lhs, list_rhs):
    """ 
    Calculates the MSE between two lists. Throws an error if the lists don't
    have the same lenght
    """
    assert(len(list_lhs) == len(list_rhs))
    return np.square(np.absolute(list_lhs - list_rhs)).mean()
    

  def frequency_objective_function(self, x):
    """
    Creates the objective-function for optimizing the frequency. The
    function assumes the input to be a ndarray, with the first value
    being the next frequency/iteration to minimize for. The algorithm 
    uses this frequency to create a theoretical signal, and returns the
    MSE wrt to the measured signal
    """
    f_k = x[0]

    x_d = signals.x_discrete(self.SNR)
    x_f = signals.x_frequency(f_k)

    Fx_d, _ = fft_estimator.M_point_fft(x_d, self.M)
    Fx_f, _ = fft_estimator.M_point_fft(x_f, self.M)

    return self.mse(np.absolute(Fx_d), np.absolute(Fx_f))


  def phase_objective_function(self, x):
    """
    Creates the objective-function for optimizing the phase. The
    function assumes the input to be a ndarray, with the first value
    being the next phase/iteration to minimize for. The algorithm 
    uses this phase to create a theoretical signal, and returns the
    MSE wrt to the measured signal
    """
    phi_k = x[0]

    x_d = signals.x_discrete(self.SNR)
    x_p = signals.x_phase(phi_k)

    return self.mse(x_d, x_p)


  def optimize_frequency_nelder_mead(self, x0):
    """ 
    Algorithm minimizing the MSE created between the theoretical and
    measured signal, in hope of estimating the frequency that is 
    embedded in the measured signal

    Returns lists of optimized frequencies, and their respective MSEs
    """
    frequencies = []

    for i in range(self.num_opt):
        results = optimize.minimize(self.frequency_objective_function, x0, method="Nelder-Mead")
        frequencies.append(results.x[0])
    
    return frequencies


  def optimize_phase_nelder_mead(self, x0):
    """ 
    Algorithm minimizing the MSE created between the theoretical and
    measured signal, in hope of estimating the phase that is 
    embedded in the measured signal

    Returns lists of optimized phases, and their respective MSEs
    """
    phases = []

    for i in range(self.num_opt):
        results = optimize.minimize(self.phase_objective_function, x0, method="Nelder-Mead")
        phases.append(results.x[0])
    
    return phases

if __name__ == '__main__':
  SNRs = [-10, 0, 10, 20, 30, 40, 50, 60]

  f0 = 150000
  phi0 = pi / 2.0

  all_frequencies = []
  all_phases = []

  avg_frequency = []
  avg_phase = []

  var_frequency = []
  var_phase = []
  
  for SNR in SNRs:
    opt = Optimize(SNR=SNR)

    frequencies = opt.optimize_frequency_nelder_mead(f0)
    # phases = opt.optimize_phase_nelder_mead(phi0)

    avg_frequency.append(np.average(frequencies))
    var_frequency.append(np.var(frequencies))

    # avg_phase.append(np.average(phases))
    # var_phase.append(np.var(phases))

    all_frequencies.append(frequencies)
    # all_phases.append(phases)
  
  plt.figure(1)
  plt.title("Scattering of frequencies for a given SNR")
  for i in range(len(SNRs)):
    y = all_frequencies[i]
    x = [SNRs[i] for k in range(len(y))]

    plt.scatter(x, y, color='green')
    # plt.scatter(SNRs[i], cfg.f0, color='red')

  plt.xlabel("SNR")
  plt.ylabel("Frequencies")
  plt.show()


  for i in range(len(SNRs)):
    plt.figure(i+2)
    plt.title("Scattering of frequencies for SNR = " + str(SNRs[i]))

    freqs = all_frequencies[i]

    freq_count = Counter(freqs)
    most_common_freqs = freq_count.most_common()

    for freq_tuple in most_common_freqs:
      x = floor(freq_tuple[0])
      y = floor(freq_tuple[1])
      plt.bar(x, y)
    
    plt.xlabel("Frequencies")
    plt.ylabel("Number of estimates")

    plt.show()