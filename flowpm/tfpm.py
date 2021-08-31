""" Core FastPM elements"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from flowpm.tfbackground import f1, E, f2, Gf, gf, gf2, D1, D2, D1f
from flowpm.utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout
from flowpm.kernels import fftk, laplace_kernel, gradient_kernel, longrange_kernel

__all__ = ['linear_field', 'lpt_init', 'nbody']


def linear_field(nc,
                 boxsize,
                 pk,
                 kvec=None,
                 batch_size=1,
                 seed=None,
                 dtype=tf.float32,
                 name="LinearField"):
  """Generates a linear field with a given linear power spectrum.
  
  Parameters:
  -----------
  nc: int, or list of ints
    Number of cells in the field. If a list is provided, number of cells per
    dimension.

  boxsize: float, or list of floats
    Physical size of the cube, in Mpc/h.

  pk: interpolator
    Power spectrum to use for the field

  kvec: array
    k_vector corresponding to the cube, optional

  batch_size: int
    Size of batches

  seed: int
    Seed to initialize the gaussian random field

  dtype: tf.dtype
    Type of the sampled field, e.g. tf.float32 or tf.float64

  Returns
  ------
  linfield: tensor (batch_size, nc, nc, nc)
    Realization of the linear field with requested power spectrum
  """
  with tf.name_scope(name):
    # Transform nc to a list of necessary
    if isinstance(nc, int):
      nc = [nc, nc, nc]
    if isinstance(boxsize, int) or isinstance(boxsize, float):
      boxsize = [boxsize, boxsize, boxsize]

    if kvec is None:
      kvec = fftk(nc, symmetric=False)
    kmesh = sum((kk / boxsize[i] * nc[i])**2 for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh)

    whitec = white_noise(nc, batch_size=batch_size, seed=seed, type='complex')
    lineark = tf.multiply(whitec, (pkmesh /
                                   (boxsize[0] * boxsize[1] * boxsize[2]))**0.5)
    linear = c2r3d(lineark, norm=nc[0] * nc[1] * nc[2], name=name, dtype=dtype)
    return linear


def lpt1(dlin_k, pos, kvec=None, name="LTP1"):
  """ Run first order LPT on linear density field, returns displacements of particles
      reading out at q. The result has the same dtype as q.

  Parameters:
  -----------
  dlin_k: TODO: @modichirag add documentation

  Returns:
  --------
  displacement: tensor (batch_size, npart, 3)
    Displacement field
  """
  with tf.name_scope(name):
    dlin_k = tf.convert_to_tensor(dlin_k, name="lineark")
    pos = tf.convert_to_tensor(pos, name="pos")

    shape = dlin_k.get_shape().as_list()
    batch_size, nc = shape[0], shape[1:]
    if kvec is None:
      kvec = fftk(nc, symmetric=False)

    lap = tf.cast(laplace_kernel(kvec), tf.complex64)

    displacement = []
    for d in range(3):
      kweight = gradient_kernel(kvec, d) * lap
      dispc = tf.multiply(dlin_k, kweight)
      disp = c2r3d(dispc, norm=nc[0] * nc[1] * nc[2])
      displacement.append(cic_readout(disp, pos))
    displacement = tf.stack(displacement, axis=2)
    return displacement


def lpt2_source(dlin_k, kvec=None, name="LPT2Source"):
  """ Generate the second order LPT source term.

  Parameters:
  -----------
  dlin_k: TODO: @modichirag add documentation

  Returns:
  --------
  source: tensor (batch_size, nc, nc, nc)
    Source term
  """
  with tf.name_scope(name):
    dlin_k = tf.convert_to_tensor(dlin_k, name="lineark")

    shape = dlin_k.get_shape()
    batch_size, nc = shape[0], shape[1:]
    if kvec is None:
      kvec = fftk(nc, symmetric=False)
    source = tf.zeros(tf.shape(dlin_k))
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []
    # diagnoal terms
    lap = tf.cast(laplace_kernel(kvec), tf.complex64)

    for d in range(3):
      grad = gradient_kernel(kvec, d)
      kweight = lap * grad * grad
      phic = tf.multiply(dlin_k, kweight)
      phi_ii.append(c2r3d(phic, norm=nc[0] * nc[1] * nc[2]))

    for d in range(3):
      source = tf.add(source, tf.multiply(phi_ii[D1[d]], phi_ii[D2[d]]))

    # free memory
    phi_ii = []

    # off-diag terms
    for d in range(3):
      gradi = gradient_kernel(kvec, D1[d])
      gradj = gradient_kernel(kvec, D2[d])
      kweight = lap * gradi * gradj
      phic = tf.multiply(dlin_k, kweight)
      phi = c2r3d(phic, norm=nc[0] * nc[1] * nc[2])
      source = tf.subtract(source, tf.multiply(phi, phi))

    source = tf.multiply(source, 3.0 / 7.)
    return r2c3d(source, norm=nc[0] * nc[1] * nc[2])


def lpt_init(cosmo, linear, a, order=2, kvec=None, name="LPTInit"):
  """ Estimate the initial LPT displacement given an input linear (real) field

  Parameters:
  -----------
  TODO: documentation
  """
  with tf.name_scope(name):
    linear = tf.convert_to_tensor(linear, name="linear")

    assert order in (1, 2)
    shape = linear.get_shape().as_list()
    batch_size, nc = shape[0], shape[1:]

    dtype = np.float32
    Q = np.indices(nc).reshape(3, -1).T.astype(dtype)
    Q = np.repeat(Q.reshape((1, -1, 3)), batch_size, axis=0)
    pos = Q

    lineark = r2c3d(linear, norm=nc[0] * nc[1] * nc[2])

    DX = tf.multiply(D1(cosmo, a), lpt1(lineark, pos))
    P = tf.multiply(a**2 * f1(cosmo, a) * E(cosmo, a), DX)
    F = tf.multiply(a**2 * E(cosmo, a) * gf(cosmo, a) / D1(cosmo, a), DX)
    if order == 2:
      DX2 = tf.multiply(D2(cosmo, a), lpt1(lpt2_source(lineark), pos))
      P2 = tf.multiply(a**2 * f2(cosmo, a) * E(cosmo, a), DX2)
      F2 = tf.multiply(a**2 * E(cosmo, a) * gf2(cosmo, a) / D2(cosmo, a), DX2)
      DX = tf.add(DX, DX2)
      P = tf.add(P, P2)
      F = tf.add(F, F2)

    X = tf.add(DX, Q)
    return tf.stack((X, P, F), axis=0)


def apply_longrange(x,
                    delta_k,
                    split=0,
                    factor=1,
                    kvec=None,
                    name="ApplyLongrange"):
  """ like long range, but x is a list of positions
  TODO: Better documentation, also better name?
  """
  # use the four point kernel to suppresse artificial growth of noise like terms
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x, name="pos")
    delta_k = tf.convert_to_tensor(delta_k, name="delta_k")

    shape = delta_k.get_shape()
    nc = shape[1:]

    if kvec is None:
      kvec = fftk(nc, symmetric=False)

    ndim = 3
    norm = nc[0] * nc[1] * nc[2]
    lap = tf.cast(laplace_kernel(kvec), tf.complex64)
    fknlrange = longrange_kernel(kvec, split)
    kweight = lap * fknlrange
    pot_k = tf.multiply(delta_k, kweight)

    f = []
    for d in range(ndim):
      force_dc = tf.multiply(pot_k, gradient_kernel(kvec, d))
      forced = c2r3d(force_dc, norm=norm)
      force = cic_readout(forced, x)
      f.append(force)

    f = tf.stack(f, axis=2)
    f = tf.multiply(f, factor)
    return f


def kick(cosmo, state, ai, ac, af, dtype=tf.float32, name="Kick", **kwargs):
  """Kick the particles given the state

  Parameters
  ----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3)

  ai, ac, af: float
  """
  with tf.name_scope(name):
    state = tf.convert_to_tensor(state, name="state")

    fac = 1 / (ac**2 * E(cosmo, ac)) * (Gf(cosmo, af) - Gf(cosmo, ai)) / gf(
        cosmo, ac)
    fac = tf.cast(fac, dtype=dtype)
    indices = tf.constant([[1]])
    update = tf.expand_dims(tf.multiply(fac, state[2]), axis=0)
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    state = tf.add(state, update)
    return state


def drift(cosmo, state, ai, ac, af, dtype=tf.float32, name="Drift", **kwargs):
  """Drift the particles given the state

  Parameters
  ----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3)

  ai, ac, af: float
  """
  with tf.name_scope(name):
    state = tf.convert_to_tensor(state, name="state")

    fac = 1. / (ac**3 * E(cosmo, ac)) * (D1(cosmo, af) - D1(cosmo, ai)) / D1f(
        cosmo, ac)
    fac = tf.cast(fac, dtype=dtype)
    indices = tf.constant([[0]])
    update = tf.expand_dims(tf.multiply(fac, state[1]), axis=0)
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    state = tf.add(state, update)
    return state


def force(cosmo,
          state,
          nc,
          pm_nc_factor=1,
          kvec=None,
          dtype=tf.float32,
          name="Force",
          **kwargs):
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
  with tf.name_scope(name):
    state = tf.convert_to_tensor(state, name="state")

    shape = state.get_shape()
    batch_size = shape[1]
    ncf = [n * pm_nc_factor for n in nc]

    rho = tf.zeros([batch_size] + ncf)
    wts = tf.ones((batch_size, nc[0] * nc[1] * nc[2]))
    nbar = nc[0] * nc[1] * nc[2] / (ncf[0] * ncf[1] * ncf[2])

    rho = cic_paint(rho, tf.multiply(state[0], pm_nc_factor), wts)
    rho = tf.multiply(rho,
                      1. / nbar)  # I am not sure why this is not needed here
    delta_k = r2c3d(rho, norm=ncf[0] * ncf[1] * ncf[2])
    fac = tf.cast(1.5 * cosmo.Omega_m, dtype=dtype)
    update = apply_longrange(
        tf.multiply(state[0], pm_nc_factor), delta_k, split=0, factor=fac)

    update = tf.expand_dims(update, axis=0) / pm_nc_factor

    indices = tf.constant([[2]])
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    mask = tf.stack((tf.ones_like(state[0]), tf.ones_like(
        state[0]), tf.zeros_like(state[0])),
                    axis=0)
    state = tf.multiply(state, mask)
    state = tf.add(state, update)
    return state


def nbody(cosmo,
          state,
          stages,
          nc,
          pm_nc_factor=1,
          return_intermediate_states=False,
          name="NBody"):
  """
  Integrate the evolution of the state across the givent stages

  Parameters:
  -----------
  cosmo: cosmology
    Cosmological parameter object

  state: tensor (3, batch_size, npart, 3)
    Input state

  stages: array
    Array of scale factors

  nc: int, or list of ints
    Number of cells

  pm_nc_factor: int
    Upsampling factor for computing

  return_intermediate_states: boolean
    If true, the frunction will return each intermediate states,
    not only the last one.

  Returns
  -------
  state: tensor (3, batch_size, npart, 3), or list of states
    Integrated state to final condition, or list of intermediate steps
  """
  with tf.name_scope(name):
    state = tf.convert_to_tensor(state, name="state")

    if isinstance(nc, int):
      nc = [nc, nc, nc]

    # Unrolling leapfrog integration to make tf Autograph happy
    if len(stages) == 0:
      return state

    ai = stages[0]

    # first force calculation for jump starting
    state = force(cosmo, state, nc, pm_nc_factor=pm_nc_factor)

    x, p, f = ai, ai, ai
    intermediate_states = []
    # Loop through the stages
    for i in range(len(stages) - 1):
      a0 = stages[i]
      a1 = stages[i + 1]
      ah = (a0 * a1)**0.5

      # Kick step
      state = kick(cosmo, state, p, f, ah)
      p = ah

      # Drift step
      state = drift(cosmo, state, x, p, a1)
      x = a1

      # Force
      state = force(cosmo, state, nc, pm_nc_factor=pm_nc_factor)
      f = a1

      # Kick again
      state = kick(cosmo, state, p, f, a1)
      p = a1
      intermediate_states.append((a1, state))

    if return_intermediate_states:
      return intermediate_states
    else:
      return state
