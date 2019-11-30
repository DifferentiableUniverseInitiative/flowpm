"""Distributed Computation of Fourier Kernels"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_bspline_kernel(x, channels, transpose=False, dtype=tf.float32, order=4):
  """Creates a 5x5x5 b-spline kernel.
  Args:
    num_channels: The number of channels of the image to filter.
    dtype: The type of an element in the kernel.
  Returns:
    A tensor of shape `[5, 5, 5, num_channels, num_channels]`.
  """
  mesh = x.mesh
  in_dim = x.shape[-1]
  num_channels = channels.size
  if order == 8:
    kernel = np.array(( 1., 8., 28., 56., 70., 56., 28., 8., 1.), dtype=dtype.as_numpy_dtype())
  elif order==2:
    kernel = np.array(( 1., 2., 1.), dtype=dtype.as_numpy_dtype())
  else:
    kernel = np.array(( 1., 4., 6., 4., 1.), dtype=dtype.as_numpy_dtype())
  size = len(kernel)
  kernel = np.einsum('ij,k->ijk', np.outer(kernel, kernel), kernel)
  kernel /= np.sum(kernel)
  kernel = kernel[:, :, :, np.newaxis, np.newaxis]
  kernel = tf.constant(kernel, dtype=dtype) * tf.eye(num_channels, dtype=dtype)

  fd_dim = mtf.Dimension("fd", size)
  fh_dim = mtf.Dimension("fh", size)
  fw_dim = mtf.Dimension("fw", size)
  if transpose:
    return mtf.import_tf_tensor(mesh, kernel, shape=[fd_dim, fh_dim, fw_dim, channels, in_dim])
  else:
    return mtf.import_tf_tensor(mesh, kernel, shape=[fd_dim, fh_dim, fw_dim, in_dim, channels])

def apply_gradient_kernel(kfield, kvec, order=1):
  """
  Computes gradients in Fourier space along all three spatial directions

  """
  # TODO: support order=0 kernel
  assert order == 1
  def _swise_fn(kfield, kx, ky, kz):
    kx = tf.reshape(kx, [1, -1, 1, 1])
    ky = tf.reshape(ky, [1, 1, -1, 1])
    kz = tf.reshape(kz, [1, 1, 1, -1])
    dkfield_dx = kfield * 1.j/6.0 * tf.cast(8 * tf.sin(kx) - tf.sin(2*kx), kfield.dtype)
    dkfield_dy = kfield * 1.j/6.0 * tf.cast(8 * tf.sin(ky) - tf.sin(2*ky), kfield.dtype)
    dkfield_dz = kfield * 1.j/6.0 * tf.cast(8 * tf.sin(kz) - tf.sin(2*kz), kfield.dtype)
    return dkfield_dx, dkfield_dy, dkfield_dz
  dkfield_dx, dkfield_dy, dkfield_dz = mtf.slicewise(_swise_fn, [kfield] + kvec,
                         output_shape=[kfield.shape]*3,
                         output_dtype=[tf.complex64]*3,
                         splittable_dims=kfield.shape[:])
  return dkfield_dx, dkfield_dy, dkfield_dz

def apply_gradient_laplace_kernel(kfield, kvec, order=1):
  """
  Computes gradients in Fourier space along all three spatial directions
  """
  # TODO: support order=0 kernel
  assert order == 1
  def _swise_fn(kfield, kx, ky, kz):
    kx = tf.reshape(kx, [1, -1, 1, 1])
    ky = tf.reshape(ky, [1, 1, -1, 1])
    kz = tf.reshape(kz, [1, 1, 1, -1])
    kk = (kx **2 + ky**2 + kz**2)
    wts = tf.where(kk>0, 1./kk, tf.zeros_like(kk))
    kfield = kfield * tf.cast(wts, kfield.dtype)
    dkfield_dx = kfield * 1.j/6.0 * tf.cast(8 * tf.sin(kx) - tf.sin(2*kx), kfield.dtype)
    dkfield_dy = kfield * 1.j/6.0 * tf.cast(8 * tf.sin(ky) - tf.sin(2*ky), kfield.dtype)
    dkfield_dz = kfield * 1.j/6.0 * tf.cast(8 * tf.sin(kz) - tf.sin(2*kz), kfield.dtype)
    return dkfield_dx, dkfield_dy, dkfield_dz
  dkfield_dx, dkfield_dy, dkfield_dz = mtf.slicewise(_swise_fn, [kfield] + kvec,
                         output_shape=[kfield.shape]*3,
                         output_dtype=[tf.complex64]*3,
                         splittable_dims=kfield.shape[:])
  return dkfield_dx, dkfield_dy, dkfield_dz

def apply_laplace_kernel(kfield, kvec):
  """
  Apply the Laplace kernel
  """
  def _cwise_fn(kfield, kx, ky, kz):
    kx = tf.reshape(kx, [1, -1, 1, 1])
    ky = tf.reshape(ky, [1, 1, -1, 1])
    kz = tf.reshape(kz, [1, 1, 1, -1])
    kk = (kx **2 + ky**2 + kz**2)
    wts = tf.where(kk>0, 1./kk, tf.zeros_like(kk))
    return kfield * tf.cast(wts, kfield.dtype)
  kfield = mtf.cwise(_cwise_fn, [kfield] + kvec,
                     output_dtype=kfield.dtype)
  return kfield

def apply_longrange_kernel(kfield, kvec, r_split):
  """
  Apply the longrange kernel
  """
  if r_split == 0:
    return kfield
  def _cwise_fn(kfield, kx, ky, kz):
    kx = tf.reshape(kx, [1, -1, 1, 1])
    ky = tf.reshape(ky, [1, 1, -1, 1])
    kz = tf.reshape(kz, [1, 1, 1, -1])
    kk = (kx **2 + ky**2 + kz**2)
    wts = tf.exp(-kk * r_split**2)
    return kfield * tf.cast(wts, kfield.dtype)
  kfield = mtf.cwise(_cwise_fn, [kfield] + kvec,
                     output_dtype=kfield.dtype)
  return kfield
