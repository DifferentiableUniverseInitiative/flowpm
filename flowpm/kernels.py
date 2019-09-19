""" Implementation of kernels required by FastPM. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
  """ Return k_vector given a shape (nc, nc, nc) and box_size
  """
  k = []
  for d in range(len(shape)):
    kd = np.fft.fftfreq(shape[d])
    kd *= 2 * np.pi / boxsize * shape[d]
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

def gradient_kernel(kvec, direction, boxsize):
  """
  Computes the gradient kernel in the requested direction

  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space

  direction: int
    Index of the direction in which to take the gradient

  boxsize: float
    Size of a cell in the 3D mesh, in Mpc/h TODO; confirm unit

  Returns:
  --------
  wts: array
    Complex kernel
  """
  nc = len(kvec[0])
  cellsize = boxsize/nc
  w = kvec[direction] * cellsize
  a = 1 / (6.0 * cellsize) * (8 * np.sin(w) - np.sin(2 * w))
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

def longrange(config, x, delta_k, r_split=0, factor=1):
  """ like long range, but x is a list of positions """
  # use the four point kernel to suppresse artificial growth of noise like terms

  ndim = 3
  norm = config['nc']**3
  lap = laplace_kernel(kvec)
  fknlrange = longrange_kernel(kvec, r_split)
  kweight = lap * fknlrange
  pot_k = tf.multiply(delta_k, kweight)

  f = []
  for d in range(ndim):
    force_dc = tf.multiply(pot_k, gradient(config, d))
    forced = c2r3d(force_dc, norm=norm)
    force = cic_readout(forced, x)
    f.append(force)

  f = tf.stack(f, axis=1)
  f = tf.multiply(f, factor)
  return f
