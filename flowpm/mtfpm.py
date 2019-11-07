""" Core FastPM elements, implemented in Mesh Tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from astropy.cosmology import Planck15

PerturbationGrowth = lambda cosmo, *args, **kwargs: MatterDominated(Omega0_lambda = cosmo.Ode0,
                                                                    Omega0_m = cosmo.Om0,
                                                                    Omega0_k = cosmo.Ok0,
                                                                    *args, **kwargs)

def linear_field(nc, boxsize, pk, batch_size=1,
                 kvec=None, seed=None, name=None, dtype=tf.float32):
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
  with tf.name_scope(name, "LinearField"):
    if kvec is None:
      kvec = fftk((nc, nc, nc), symmetric=False)
    kmesh = sum((kk / boxsize * nc)**2 for kk in kvec)**0.5
    pkmesh = pk(kmesh)

    whitec = white_noise(nc, batch_size=batch_size, seed=seed, type='complex')
    lineark = tf.multiply(whitec, (pkmesh/boxsize**3)**0.5)
    linear = c2r3d(lineark, norm=nc**3, name=name, dtype=dtype)
    return linear
