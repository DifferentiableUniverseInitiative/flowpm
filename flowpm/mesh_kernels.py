"""Distributed Computation of Fourier Kernels"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow as tf

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
