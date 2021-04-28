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
