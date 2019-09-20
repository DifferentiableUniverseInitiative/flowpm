""" Implementation of kernels required by FastPM. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def fftk(shape, symmetric=True, finite=False, dtype=np.float64):
  """ Return k_vector given a shape (nc, nc, nc) and box_size
  """
  k = []
  for d in range(len(shape)):
    kd = np.fft.fftfreq(shape[d])
    kd *= 2 * np.pi
    kdshape = np.ones(len(shape), dtype='int')
    if symmetric and d == len(shape) -1:
        kd = kd[:shape[d]//2 + 1]
    kdshape[d] = len(kd)
    kd = kd.reshape(kdshape)

    k.append(kd.astype(dtype))
  del kd, kdshape
  return k

def laplace_kernel(kvec):
  """
  Compute the Laplace kernel from a given K vector

  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space

  Returns:
  --------
  wts: array
    Complex kernel
  """
  kk = sum(ki**2 for ki in kvec)
  mask = (kk == 0).nonzero()
  kk[mask] = 1
  wts = 1/kk
  imask = (~(kk==0)).astype(int)
  wts *= imask
  return wts

def gradient_kernel(kvec, direction, order=0):
  """
  Computes the gradient kernel in the requested direction

  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space

  direction: int
    Index of the direction in which to take the gradient

  Returns:
  --------
  wts: array
    Complex kernel
  """
  if order == 0:
    return 1j * kvec[direction]
  else:
    nc = len(kvec[0])
    w = kvec[direction]
    a = 1 / 6.0  * (8 * np.sin(w) - np.sin(2 * w))
    wts = a*1j
    return wts

def longrange_kernel(kvec, r_split):
  """
  Computes a long range kernel

  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space

  r_split: float
    TODO: @modichirag add documentation

  Returns:
  --------
  wts: array
    kernel
  """
  if r_split != 0:
    kk = sum(ki ** 2 for ki in kvec)
    return np.exp(-kk * r_split**2)
  else:
    return 1.
