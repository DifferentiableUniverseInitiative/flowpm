""" Core FastPM elements, implemented in Mesh Tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from astropy.cosmology import Planck15

import mesh_tensorflow as mtf

from .background import MatterDominated
from . import mesh_ops
from . import mesh_utils
from . import mesh_kernels

PerturbationGrowth = lambda cosmo, *args, **kwargs: MatterDominated(Omega0_lambda = cosmo.Ode0,
                                                                    Omega0_m = cosmo.Om0,
                                                                    Omega0_k = cosmo.Ok0,
                                                                    *args, **kwargs)

def linear_field(mesh, shape, boxsize, pk, kvec, seed=None, dtype=tf.float32):
  """Generates a linear field with a given linear power spectrum

  Parameters:
  -----------
  nc: int
    Number of cells in the field

  boxsize: float
    Physical size of the cube, in Mpc/h TODO: confirm units

  pk: [batch size]
    Power spectrum to use for the field

  kvec: array
    k_vector corresponding to the cube, optional

  Returns
  ------
  linfield: tensor (batch_size, nc, nc, nc)
    Realization of the linear field with requested power spectrum
  """
  x_dim, y_dim, z_dim = shape[-3:]
  nc = z_dim.size
  field = mesh_ops.random_normal(mesh, shape=shape,
                                 mean=0, stddev=nc**1.5, dtype=tf.float32)
  kfield = mesh_utils.r2c3d(field)

  # Element-wise function that applies a Fourier kernel
  def _cwise_fn(kfield, pk, kx, ky, kz):
    kx = tf.reshape(kx, [-1, 1, 1])
    ky = tf.reshape(ky, [1, -1, 1])
    kz = tf.reshape(kz, [1, 1, -1])
    kk = tf.sqrt((kx / boxsize * nc)**2 + (ky/ boxsize * nc)**2 + (kz/ boxsize * nc)**2)
    shape = kk.shape
    kk = tf.reshape(kk, [-1])
    pkmesh = tfp.math.interp_regular_1d_grid(x=kk, x_ref_min=1e-05, x_ref_max=1000.0,
                                             y_ref=pk, grid_regularizing_transform=tf.log)
    pkmesh = tf.reshape(pkmesh, shape)
    kfield = kfield * tf.cast((pkmesh/boxsize**3)**0.5, tf.complex64)
    return kfield
  kfield = mtf.cwise(_cwise_fn,
                     [kfield, pk] + kvec,
                     output_dtype=tf.complex64)
  return mesh_utils.c2r3d(kfield)

def lpt1(kfield, pos, kvec, splitted_dims, nsplits):
  # First apply Laplace Kernel
  grad_kfield = mesh_kernels.apply_gradient_laplace_kernel(kfield, kvec)
  # Now apply gradient kernel
  # grad_kfield = mesh_kernels.apply_gradient_kernel(kfield, kvec)
  # Compute displacements on mesh
  displacement = [ mesh_utils.c2r3d(f) for f in grad_kfield ]
  # Readout to particle positions
  displacement = mtf.stack([ mesh_utils.cic_readout(d, pos, splitted_dims, nsplits) for d in displacement],"ndim",axis=4)
  return displacement

def lpt_init(field, a0, kvec, splitted_dims, nsplits, order=1, cosmology=Planck15):
  a = a0
  batch_dim, x_dim, y_dim, z_dim = field.shape

  # Create particles on uniform grid
  mstate = mesh_ops.mtf_indices(field.mesh, shape=[x_dim, y_dim, z_dim], dtype=tf.float32)
  X = mtf.einsum([mtf.ones(field.mesh, [batch_dim]), mstate], output_shape=[batch_dim] + mstate.shape[:])

  # Computes Fourier Transform of input field
  kfield = mesh_utils.r2c3d(field)

  pt = PerturbationGrowth(cosmology, a=[a], a_normalize=1.0)

  DX = pt.D1(a) * lpt1(kfield, X, kvec, splitted_dims, nsplits)
  P = (a ** 2 * pt.f1(a) * pt.E(a)) * DX
  F = (a ** 2 * pt.E(a) * pt.gf(a) / pt.D1(a)) * DX
  # TODO: Implement 2nd order LPT

  # Moves the particles according to displacement
  X = X + DX

  return X, P, F
