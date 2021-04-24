#!/usr/bin/env python

import cfg
import statistics
import numpy as np
import matplotlib.pyplot as plt
import Signals
import estimation
import error_calculation

from scipy import optimize, fft, fftpack
from math import pi


class Optimize:
  def __init__(self, f0=None, phi0=None, M=None, SNR=None):
    self.N = cfg.N
    self.T = cfg.Ts
    self.n0 = cfg.n0
    self.Fs = cfg.Fs
    self.num_opt = cfg.num_optimizations

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

    x_d = signals.x_discrete(self.SNR_dB)
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

    x_d = signals.x_discrete(self.SNR_dB)
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
    mse = []

    for i in range(self.num_opt):
        results = optimize.minimize(self.frequency_objective_function, x0, method="Nelder-Mead")
        frequencies.append(results.x[0])
        mse.append((self.f0 - results.x[0])**2)
    
    return frequencies, mse


  def optimize_phase_nelder_mead(self, x0):
    """ 
    Algorithm minimizing the MSE created between the theoretical and
    measured signal, in hope of estimating the phase that is 
    embedded in the measured signal

    Returns lists of optimized phases, and their respective MSEs
    """
    phases = []
    mse = []

    for i in range(self.num_opt):
        results = optimize.minimize(self.phase_objective_function, x0, method="Nelder-Mead")
        phases.append(results.x[0])
        mse.append((self.phi0 - results.x[0])**2)
    
    return phases, mse
