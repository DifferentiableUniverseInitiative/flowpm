import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

from pmesh.pm import ParticleMesh
from fastpm.core import Solver as Solver

import flowpm.tfpm as tfpm

np.random.seed(0)

def test_linear_field_shape():
  bs = 50
  nc = 16

  klin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)

  with tf.Session() as sess:
    field = tfpm.linear_field(nc, bs, ipklin, batch_size=5)
    sess.run(tf.global_variables_initializer())
    tfread = sess.run(field)

  assert tfread.shape == (5, 16, 16, 16)

def test_lpt_init():
  bs = 50
  nc = 16
  a0 = 0.1
  from nbodykit.cosmology import Cosmology, EHPower, Planck15

  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
  solver = Solver(pm, Planck15)

  # Generate initial state with fastpm
  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)
  statelpt = solver.lpt(lineark, grid, a0, order=2)

  # Same thing with flowpm
  with tf.Session() as sess:
    tlinear = tf.expand_dims(tf.constant(lineark.c2r()), 0)
    tflptic = tfpm.lpt_init(tlinear, bs, a0, order=2)

    sess.run(tf.global_variables_initializer())
    tfread = sess.run(tflptic)

  assert_allclose(statelpt.X, tfread[0,0], rtol=1e-7)
