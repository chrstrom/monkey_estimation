#!/usr/bin/env python

import numpy as np

def mse(list_lhs, list_rhs):
  # Finds the mean-square-error between two signals
  assert(len(list_lhs) == len(list_rhs))
  return np.square(np.absolute(list_lhs - list_rhs)).mean()