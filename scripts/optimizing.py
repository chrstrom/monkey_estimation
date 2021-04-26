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


def calculate_optimal_frequency(frequencies):
  """ 
  This algorithm is an attempt in calculating the optimal frequency

  This algorithm will attempt to identify the frequency which occurs 
  the most. To prevent any decimal-values for affecting the results, the
  algorithm floors the input. Therefore the function assumes the input to 
  have frequencies >= 1 Hz

  The algorithm will return the floored frequency that occurs the most

  If all frequencies are different, the algorithm will use the sample
  mean
  """
  floored_frequencies = [floor(freq) for freq in frequencies]
  counted_frequencies = Counter(floored_frequencies)
  
  most_common_freq = counted_frequencies.most_common(1)

  if most_common_freq[0][1] > 1:
    return most_common_freq[0][0]
  
  # All frequencies are unique
  return np.mean(floored_frequencies)

def mse(list_lhs, list_rhs):
  """ 
  Calculates the MSE between two lists. Throws an error if the lists don't
  have the same lenght
  """
  assert(len(list_lhs) == len(list_rhs))
  return np.square(np.absolute(list_lhs - list_rhs)).mean()
  

def frequency_objective_function(x, M, x_d, phi_hat):
  """
  Creates the objective-function for optimizing the frequency. The
  function assumes the input to be a ndarray, with the first value
  being the next frequency/iteration to minimize for. The algorithm 
  uses this frequency to create a theoretical signal, and returns the
  MSE wrt to the measured signal
  """

  f_k = x[0]
  x_f = signals.x_ideal(f_k, phi_hat) # Phase has no effect as it removed through FFT

  Fx_d, _ = fft_estimator.M_point_fft(x_d, M)
  Fx_f, _ = fft_estimator.M_point_fft(x_f, M)

  return mse(np.absolute(Fx_d), np.absolute(Fx_f))


def phase_objective_function(x, x_d, f_hat):
  """
  Creates the objective-function for optimizing the phase. The
  function assumes the input to be a ndarray, with the first value
  being the next phase/iteration to minimize for. The algorithm 
  uses this phase to create a theoretical signal, and returns the
  MSE wrt to the measured signal
  """
  phi_k = x[0]

  x_p = signals.x_ideal(f_hat, phi_k)

  return mse(x_d, x_p)

"""
if __name__ == '__main__':
  SNRs = [-10]#, 0, 10, 20, 30, 40, 50, 60]

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
    optimal_frequency = calculate_optimal_frequency(frequencies)
    opt.set_f_hat(optimal_frequency)
    phases = opt.optimize_phase_nelder_mead(phi0)

    avg_frequency.append(np.average(frequencies))
    var_frequency.append(np.var(frequencies))

    avg_phase.append(np.average(phases))
    var_phase.append(np.var(phases))

    all_frequencies.append(frequencies)
    all_phases.append(phases)
  
  plt.figure(1)
  plt.title("Scattering of frequencies for a given SNR")
  for i in range(len(SNRs)):
    y = all_frequencies[i]
    x = [SNRs[i] for k in range(len(y))]

    plt.scatter(x, y, color='green')
    plt.scatter(SNRs[i], f0, color='red')

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
"""