#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:19:45 2021

@author: Denise Lanzieri
"""

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
  tf.config.experimental.set_memory_growth(device, True)
import numpy as np

import sys
import flowpm
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from flowpm.utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout
from flowpm.kernels import fftk
from numpy.testing import assert_allclose
import time

#%%
field = 5.
box_size = 100.
nc = 32
Omega_c = 0.2589
sigma8 = 0.8159
nsteps = 5
stages = np.linspace(0.1, 1., nsteps, endpoint=True)
batch_size = 1
seed = 100

#%%


def whitenoise_to_linear(nc,
                         boxsize,
                         whitec,
                         pk,
                         kvec=None,
                         batch_size=1,
                         seed=None,
                         dtype=tf.float32,
                         name="LinearField"):
  """Generates a linear field with a given linear power spectrum and whitenoise realization
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

    lineark = tf.multiply(whitec, (pkmesh /
                                   (boxsize[0] * boxsize[1] * boxsize[2]))**0.5)
    linear = c2r3d(lineark, norm=nc[0] * nc[1] * nc[2], name=name, dtype=dtype)
    return linear


#


@tf.function
def val_and_grad(Omega_c, whitec):
  params = tf.stack([Omega_c])
  with tf.GradientTape() as tape:
    tape.watch(params)
    cosmology = flowpm.cosmology.Planck15(Omega_c=params[0])
    k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
    pk = tfpower.linear_matter_power(cosmology, k)
    pk_fun = lambda x: tf.cast(
        tf.reshape(
            interpolate.interp_tf(
                tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.
        complex64)
    initial_conditions = whitenoise_to_linear([nc, nc, nc],
                                              [box_size, box_size, box_size],
                                              whitec,
                                              pk_fun,
                                              seed=100,
                                              batch_size=1)

    state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

    final_state = flowpm.nbody(cosmology, state, stages, [nc, nc, nc])
    final_field = flowpm.cic_paint(
        tf.zeros_like(initial_conditions), final_state[0])
    final_field = tf.reshape(final_field, [nc, nc, nc])
    loss = tf.reduce_mean(final_field**2)
  return loss, tape.gradient(loss, params)


#%%

if __name__ == "__main__":

  omegac = tf.constant(Omega_c)
  whitec = white_noise(nc, batch_size=batch_size, seed=seed, type='complex')

  #analytic derivative
  val, grad = val_and_grad(omegac, whitec)

  #Numerical derivative
  x0, x1 = omegac * 0.999, omegac * 1.001
  v0, _ = val_and_grad(x0, whitec)
  v1, _ = val_and_grad(x1, whitec)

  print('analytic derivative : ', grad)
  print('numerical derivative : ', (v0 - v1) / (x0 - x1))
  print('% difference in gradients : ',
        abs((v0 - v1) / (x0 - x1) - grad) * 100 / grad)
