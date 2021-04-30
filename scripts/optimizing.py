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

  omega_k = x[0]
  x_f = signals.x_ideal(omega_k, phi_hat) # Phase has no effect as it removed through FFT

  Fx_d, _ = fft_estimator.M_point_fft(x_d, M)
  Fx_f, _ = fft_estimator.M_point_fft(x_f, M)

  return mse(np.absolute(Fx_d), np.absolute(Fx_f))


def phase_objective_function(x, x_d, omega_hat):
  """
  Creates the objective-function for optimizing the phase. The
  function assumes the input to be a ndarray, with the first value
  being the next phase/iteration to minimize for. The algorithm 
  uses this phase to create a theoretical signal, and returns the
  MSE wrt to the measured signal
  """
  phi_k = x[0]

  x_p = signals.x_ideal(omega_hat, phi_k)

  return mse(x_d, x_p)
