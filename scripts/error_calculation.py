#!/usr/bin/env python

import numpy as np

def mse(list_lhs, list_rhs):
  # Finds the mean-square-error between two signals
  assert(len(list_lhs) == len(list_rhs))

  list_length = len(list_lhs)

  total = 0
  for i in range(list_length):
      total += (np.absolute(list_lhs[i] - list_rhs[i]))**2

  return total / list_length