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
                 post_filtering=True, downsampling_factor=2, antialias=True,
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
    low = mesh_utils.downsample(field, downsampling_factor, antialias=antialias)
  else:
    low, high = mesh_utils.split_scales(field, downsampling_factor, antialias=antialias)
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

  if return_random_field:
    return low, high, random_field
  else:
    return low, high

def lpt_init(lr_field, hr_field, a0,kvec_lr, kvec_hr, halo_size,hr_shape, lr_shape,
              part_shape,
              antialias=True, downsampling_factor=2,
             order=1, post_filtering=True, cosmology=Planck15):
  a = a0
  batch_dim = hr_field.shape[0]
  lnc = lr_shape[-1].size

  # Create particles on the high resolution grid
  mstate = mesh_ops.mtf_indices(hr_field.mesh, shape=part_shape, dtype=tf.float32)
  X = mtf.einsum([mtf.ones(hr_field.mesh, [batch_dim]), mstate], output_shape=[batch_dim] + mstate.shape[:])

  lr_kfield = mesh_utils.r2c3d(lr_field)
  hr_kfield = mesh_utils.slicewise_r2c3d(hr_field)

  grad_kfield_lr = mesh_kernels.apply_gradient_laplace_kernel(lr_kfield, kvec_lr)
  grad_kfield_hr = mesh_kernels.apply_gradient_laplace_kernel(hr_kfield, kvec_hr)

  displacement = []
  for f,g in zip(grad_kfield_lr,grad_kfield_hr):
      f = mesh_utils.c2r3d(f)
      f = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),
                        [f],
                        output_dtype=tf.float32,
                        output_shape=mtf.Shape(hr_shape[0:3]+[
                                                mtf.Dimension('sx_block', lnc//hr_shape[1].size),
                                                mtf.Dimension('sy_block', lnc//hr_shape[2].size),
                                                mtf.Dimension('sz_block', lnc)]),
                        name='my_reshape',
                        splittable_dims=lr_shape[:-1]+hr_shape[1:3]+part_shape[1:3])
      for block_size_dim in hr_shape[-3:]:
          f = mtf.pad(f, [halo_size//2**downsampling_factor, halo_size//2**downsampling_factor], block_size_dim.name)
      for blocks_dim, block_size_dim in zip(hr_shape[1:3]+[hr_shape[2]], f.shape[-3:]):
          f = mesh_ops.halo_reduce(f, blocks_dim, block_size_dim, halo_size//2**downsampling_factor)
      f = mtf.reshape(f, f.shape+[mtf.Dimension('h_dim', 1)])
      f = mesh_utils.upsample(f, downsampling_factor)
      f = mtf.reshape(f, f.shape[:-1])

      g = mesh_utils.slicewise_c2r3d(g)
      high_shape = g.shape
      # And now we remove the large scales
      g = mtf.reshape(g, g.shape+[mtf.Dimension('h_dim', 1)])
      _low = mesh_utils.downsample(g, downsampling_factor, antialias=antialias)
      g = g - mtf.reshape(mesh_utils.upsample(_low, downsampling_factor), g.shape)
      g = mtf.reshape(g, high_shape)

      d =  mesh_utils.cic_readout(f+g, X, halo_size)
      displacement.append(d)

  # Readout to particle positions
  displacement = mtf.stack([ d for d in displacement],"ndim",axis=4)

  pt = PerturbationGrowth(cosmology, a=[a], a_normalize=1.0)
  DX = pt.D1(a) * displacement
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

def force(state, lr_shape, hr_shape, kvec_lr, kvec_hr, halo_size, cosmology=Planck15,
          downsampling_factor=2, pm_nc_factor=1, antialias=True, **kwargs):
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
  lnc = lr_shape[-1].size
  part_shape = X.shape

  # Paint the particles on the high resolution mesh
  field = mtf.zeros(X.mesh, shape=hr_shape)
  for block_size_dim in hr_shape[-3:]:
    field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)
  field = mesh_utils.cic_paint(field, X, halo_size)
  for blocks_dim, block_size_dim in zip(hr_shape[1:3]+[hr_shape[2]], field.shape[-3:]):
    field = mesh_ops.halo_reduce(field, blocks_dim, block_size_dim, halo_size)

  # Split the field into low and high resolution
  field = mtf.reshape(field, field.shape+[mtf.Dimension('h_dim', 1)])
  high = field
  low = mesh_utils.downsample(field, downsampling_factor, antialias=True)
  low = mtf.reshape(low, low.shape[:-1])
  hr_field = mtf.reshape(high, high.shape[:-1])
  for block_size_dim in hr_shape[-3:]:
    low = mtf.slice(low, halo_size//2**downsampling_factor, block_size_dim.size//2**downsampling_factor, block_size_dim.name)
  lr_field = mtf.reshape(low, lr_shape)

  lr_kfield = mesh_utils.r2c3d(lr_field)
  hr_kfield = mesh_utils.slicewise_r2c3d(hr_field)

  kfield_lr = mesh_kernels.apply_longrange_kernel(lr_kfield, kvec_lr, r_split=0)
  kfield_lr = mesh_kernels.apply_gradient_laplace_kernel(lr_kfield, kvec_lr)
  kfield_hr = mesh_kernels.apply_longrange_kernel(hr_kfield, kvec_hr, r_split=0)
  kfield_hr = mesh_kernels.apply_gradient_laplace_kernel(kfield_hr, kvec_hr)

  displacement = []
  for f,g in zip(kfield_lr, kfield_hr):
      f = mesh_utils.c2r3d(f)
      f = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),
                        [f],
                        output_dtype=tf.float32,
                        output_shape=mtf.Shape(hr_shape[0:3]+[
                                                mtf.Dimension('sx_block', lnc//hr_shape[1].size),
                                                mtf.Dimension('sy_block', lnc//hr_shape[2].size),
                                                mtf.Dimension('sz_block', lnc)]),
                        name='my_reshape',
                        splittable_dims=lr_shape[:-1]+hr_shape[1:3]+part_shape[1:3])
      for block_size_dim in hr_shape[-3:]:
          f = mtf.pad(f, [halo_size//2**downsampling_factor, halo_size//2**downsampling_factor], block_size_dim.name)
      for blocks_dim, block_size_dim in zip(hr_shape[1:3]+[hr_shape[2]], f.shape[-3:]):
          f = mesh_ops.halo_reduce(f, blocks_dim, block_size_dim, halo_size//2**downsampling_factor)
      f = mtf.reshape(f, f.shape+[mtf.Dimension('h_dim', 1)])
      f = mesh_utils.upsample(f, downsampling_factor)
      f = mtf.reshape(f, f.shape[:-1])

      g = mesh_utils.slicewise_c2r3d(g)
      high_shape = g.shape
      # And now we remove the large scales
      g = mtf.reshape(g, g.shape+[mtf.Dimension('h_dim', 1)])
      _low = mesh_utils.downsample(g, downsampling_factor, antialias=antialias)
      g = g - mtf.reshape(mesh_utils.upsample(_low, downsampling_factor), g.shape)
      g = mtf.reshape(g, high_shape)

      d =  mesh_utils.cic_readout(f+g, X, halo_size)
      displacement.append(d)

  # Readout the force to particle positions
  F = mtf.stack([ d for d in displacement],"ndim",axis=4)

  F = F * 1.5 * cosmology.Om0
  return X, P, F

def nbody(state, stages, lr_shape, hr_shape, kvec_lr, kvec_hr, halo_size, cosmology=Planck15,
          pm_nc_factor=1, downsampling_factor=2):
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
  state = force(state, lr_shape, hr_shape, kvec_lr, kvec_hr, halo_size,
                pm_nc_factor=pm_nc_factor, cosmology=cosmology,
                downsampling_factor=downsampling_factor)

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
    state = force(state, lr_shape, hr_shape, kvec_lr, kvec_hr, halo_size,
                  pm_nc_factor=pm_nc_factor, cosmology=cosmology,
                  downsampling_factor=downsampling_factor)
    f = a1

    # Kick again
    state = kick(state, p, f, a1, cosmology=cosmology)
    p = a1

  return state
