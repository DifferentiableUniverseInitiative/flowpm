import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline


from nbodykit.cosmology import Cosmology, EHPower, Planck15
from pmesh.pm import ParticleMesh
from fastpm.core import Solver as Solver
import fastpm.force.lpt as fpmops

import flowpm.tfpm as tfpm
import flowpm.utils as pmutils

np.random.seed(0)

bs = 50
nc = 16

def test_linear_field_shape():
  klin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)

  with tf.Session() as sess:
    field = tfpm.linear_field(nc, bs, ipklin, batch_size=5)

    tfread = sess.run(field)

  assert tfread.shape == (5, 16, 16, 16)

def test_lpt_init():
  a0 = 0.1

  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
  solver = Solver(pm, Planck15, B=1)

  # Generate initial state with fastpm
  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)
  statelpt = solver.lpt(lineark, grid, a0, order=2)

  # Same thing with flowpm
  with tf.Session() as sess:
    tlinear = tf.expand_dims(tf.constant(lineark.c2r()), 0)
    tflptic = tfpm.lpt_init(tlinear, bs, a0, order=2)

    tfread = sess.run(tflptic)

  assert_allclose(statelpt.X, tfread[0,0], rtol=1e-7)

def test_lpt1():
  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)

  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

  # Compute lpt1 from fastpm
  lpt = fpmops.lpt1(lineark, grid)

  # Same thing from tensorflow
  with tf.Session() as sess:
    state = tfpm.lpt1(pmutils.r2c3d(tf.expand_dims(tf.constant(lineark.c2r()), axis=0)), grid.reshape((1, -1, 3)), bs)
    tfread = sess.run(state)

  assert_allclose(lpt, tfread[0], rtol=1e-7)
