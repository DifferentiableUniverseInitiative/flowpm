""" Core FastPM elements, implemented in Mesh Tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

def linear_field(mesh, hr_shape, lr_shape,
                 boxsize, nc, pk, kvec_lr, kvec_hr, halo_size,
                 post_filtering=True, downsampling_factor=2,
                 seed=None, dtype=tf.float32, return_random_field=False):
  """Generates a linear field with a given linear power spectrum
  """
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

  # Generates the random field
  random_field = mesh_ops.random_normal(mesh, shape=hr_shape,
                                        mean=0, stddev=nc**1.5,
                                        dtype=tf.float32)
  field = random_field
  # Apply padding and perform halo exchange with neighbors
  # TODO: Figure out how to deal with the tensor size limitations
  for block_size_dim in hr_shape[-3:]:
    field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)
  for blocks_dim, block_size_dim in zip(hr_shape[1:3]+[hr_shape[2]] , field.shape[-3:]):
    field = mesh_ops.halo_reduce(field, blocks_dim, block_size_dim, halo_size)

  # We have two strategies to separate scales, before or after filtering
  field = mtf.reshape(field, field.shape+[mtf.Dimension('h_dim', 1)])
  if post_filtering:
    high = field
    low = mesh_utils.downsample(field, downsampling_factor, antialias=True)
  else:
    low, high = mesh_utils.split_scales(field, downsampling_factor, antialias=True)
  low = mtf.reshape(low, low.shape[:-1])
  high = mtf.reshape(high, high.shape[:-1])

  # Remove padding and redistribute the low resolution cube accross processes
  for block_size_dim in hr_shape[-3:]:
    low = mtf.slice(low, halo_size//2**downsampling_factor, block_size_dim.size//2**downsampling_factor, block_size_dim.name)
  low_hr_shape = low.shape
  low = mtf.reshape(low, lr_shape)

  # Apply power spectrum on both grids
  klow = mesh_utils.r2c3d(low)
  khigh = mesh_utils.r2c3d(high)
  klow = mtf.cwise(_cwise_fn, [klow, pk] + kvec_lr, output_dtype=tf.complex64)
  khigh = mtf.cwise(_cwise_fn, [khigh, pk] + kvec_hr, output_dtype=tf.complex64)
  low = mesh_utils.c2r3d(klow)
  high = mesh_utils.c2r3d(khigh)

  # Now we need to resplit the low resolution into local arrays
  low = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),
                      [low],
                      output_dtype=tf.float32,
                      output_shape=low_hr_shape,
                      name='my_reshape',
                      splittable_dims=low.shape[:-1]+low_hr_shape[1:3])

  # We reapply padding and upsample
  for block_size_dim in hr_shape[-3:]:
    low = mtf.pad(low, [halo_size//2**downsampling_factor, halo_size//2**downsampling_factor], block_size_dim.name)
  low = mtf.reshape(low, low.shape+[mtf.Dimension('h_dim', 1)])
  low = mtf.reshape(mesh_utils.upsample(low, downsampling_factor), high.shape)

  # And to make sure everything is ok, we perform a halo exchange
  for blocks_dim, block_size_dim in zip(hr_shape[1:3]+[hr_shape[2]], low.shape[-3:]):
    low = mesh_ops.halo_reduce(low, blocks_dim, block_size_dim, halo_size)

  if post_filtering:
    high = mtf.reshape(high, high.shape+[mtf.Dimension('h_dim', 1)])
    _low = mesh_utils.downsample(high, downsampling_factor)
    high = high - mtf.reshape(mesh_utils.upsample(_low, downsampling_factor), high.shape)
    high = mtf.reshape(high, low.shape)

  # Combining the two components
  field = high + low

  # All done, we now just need to remove the padding
  for block_size_dim in hr_shape[-3:]:
    field = mtf.slice(field, halo_size, block_size_dim.size, block_size_dim.name)

  if return_random_field:
    return field, random_field
  else:
    return field

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

def kick(state, ai, ac, af, cosmology=Planck15, **kwargs):
  """Kick the particles given the state

  Parameters
  ----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3)

  ai, ac, af: float
  """
  X, P, F = state

  pt = PerturbationGrowth(cosmology, a=[ai, ac, af], a_normalize=1.0)
  fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
  P += fac * F
  return X, P ,F

def drift(state, ai, ac, af, cosmology=Planck15, **kwargs):
  """Drift the particles given the state

  Parameters
  ----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3)

  ai, ac, af: float
  """
  X, P, F = state
  pt = PerturbationGrowth(cosmology, a=[ai, ac, af], a_normalize=1.0)
  fac = 1. / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
  X += fac * P
  return X, P, F

def force(state, shape, kvec, splitted_dims, nsplits, cosmology=Planck15, pm_nc_factor=1, **kwargs):
  """
  Estimate force on the particles given a state.

  Parameters:
  -----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3)

  boxsize: float
    Size of the simulation volume (Mpc/h) TODO: check units

  cosmology: astropy.cosmology
    Cosmology object

  pm_nc_factor: int
    TODO: @modichirag please add doc
  """
  X, P, F = state
  #TODO: support different factor
  assert pm_nc_factor ==1
  kfield = mesh_utils.r2c3d(mesh_utils.cic_paint(mtf.zeros(X.mesh,shape), X, splitted_dims, nsplits))

  # use the four point kernel to suppresse artificial growth of noise like terms
  kfield = mesh_kernels.apply_longrange_kernel(kfield, kvec, r_split=0)
  kforces = mesh_kernels.apply_gradient_laplace_kernel(kfield, kvec)
  F = mtf.stack([mesh_utils.cic_readout(mesh_utils.c2r3d(f), X, splitted_dims, nsplits) for f in kforces], "ndim", axis=4)

  F = F * 1.5 * cosmology.Om0
  return X, P, F

def nbody(state, stages, shape, kvec, splitted_dims, nsplits, cosmology=Planck15, pm_nc_factor=1):
  """
  Integrate the evolution of the state across the givent stages

  Parameters:
  -----------
  state: tensor (3, batch_size, npart, 3)
    Input state

  stages: array
    Array of scale factors

  nc: int
    Number of cells

  pm_nc_factor: int
    Upsampling factor for computing

  Returns
  -------
  state: tensor (3, batch_size, npart, 3)
    Integrated state to final condition
  """
  assert pm_nc_factor == 1

  # Unrolling leapfrog integration to make tf Autograph happy
  if len(stages) == 0:
    return state

  ai = stages[0]

  # first force calculation for jump starting
  state = force(state, shape, kvec, splitted_dims, nsplits, pm_nc_factor=pm_nc_factor, cosmology=cosmology)

  x, p, f = ai, ai, ai
  # Loop through the stages
  for i in range(len(stages) - 1):
    a0 = stages[i]
    a1 = stages[i + 1]
    ah = (a0 * a1) ** 0.5

    # Kick step
    state = kick(state, p, f, ah, cosmology=cosmology)
    p = ah

    # Drift step
    state = drift(state, x, p, a1, cosmology=cosmology)
    x = a1

    # Force
    state = force(state, shape, kvec, splitted_dims, nsplits, pm_nc_factor=pm_nc_factor, cosmology=cosmology)
    f = a1

    # Kick again
    state = kick(state, p, f, a1, cosmology=cosmology)
    p = a1

  return state
