import tensorflow as tf

import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

from nbodykit.cosmology import Cosmology, EHPower, Planck15
from pmesh.pm import ParticleMesh
from fastpm.core import Solver as Solver
import fastpm.force.lpt as fpmops
from fastpm.core import leapfrog

import flowpm.tfpm as tfpm
import flowpm.utils as pmutils

np.random.seed(0)

bs = 50
nc = 16

def test_linear_field_shape():
  """ Testing just the shape of the sampled linear field
  """
  klin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)

  tfread = tfpm.linear_field(nc, bs, ipklin, batch_size=5).numpy()
  assert tfread.shape == (5, 16, 16, 16)

def test_lpt_init():
  """
  Checking lpt init
  """
  a0 = 0.1

  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
  solver = Solver(pm, Planck15, B=1)

  # Generate initial state with fastpm
  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)
  statelpt = solver.lpt(lineark, grid, a0, order=1)

  # Same thing with flowpm
  tlinear = tf.expand_dims(np.array(lineark.c2r()), 0)
  tfread = tfpm.lpt_init(tlinear, a0, order=1).numpy()

  assert_allclose(statelpt.X, tfread[0,0]*bs/nc, rtol=1e-2)

def test_lpt1():
  """ Checking lpt1, this also checks the laplace and gradient kernels
  """
  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)

  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

  # Compute lpt1 from fastpm with matching kernel order
  lpt = fpmops.lpt1(lineark, grid)

  # Same thing from tensorflow
  tfread = tfpm.lpt1(pmutils.r2c3d(tf.expand_dims(np.array(lineark.c2r()), axis=0)), grid.reshape((1, -1, 3))*nc/bs).numpy()

  assert_allclose(lpt, tfread[0]*bs/nc, atol=1e-5)

def test_lpt1_64():
  """ Checking lpt1, this also checks the laplace and gradient kernels
  This variant of the test checks that it works for cubes of size 64
  """
  nc = 64
  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)

  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

  # Compute lpt1 from fastpm with matching kernel order
  lpt = fpmops.lpt1(lineark, grid)

  # Same thing from tensorflow
  tfread = tfpm.lpt1(pmutils.r2c3d(tf.expand_dims(np.array(lineark.c2r()), axis=0)), grid.reshape((1, -1, 3))*nc/bs).numpy()

  assert_allclose(lpt, tfread[0]*bs/nc, atol=5e-5)

def test_lpt2():
  """ Checking lpt2_source, this also checks the laplace and gradient kernels
  """
  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)

  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

  # Compute lpt1 from fastpm with matching kernel order
  source = fpmops.lpt2source(lineark).c2r()

  # Same thing from tensorflow
  tfsource = tfpm.lpt2_source(pmutils.r2c3d(tf.expand_dims(np.array(lineark.c2r()), axis=0)))
  tfread = pmutils.c2r3d(tfsource).numpy()

  assert_allclose(source, tfread[0], atol=1e-5)

def test_nody():
  """ Checking end to end nbody
  """
  a0 = 0.1

  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
  solver = Solver(pm, Planck15, B=1)
  stages = np.linspace(0.1, 1.0, 10, endpoint=True)

  # Generate initial state with fastpm
  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)
  statelpt = solver.lpt(lineark, grid, a0, order=1)
  finalstate = solver.nbody(statelpt, leapfrog(stages))
  final_cube = pm.paint(finalstate.X)

  # Same thing with flowpm
  tlinear = tf.expand_dims(np.array(lineark.c2r()), 0)
  state = tfpm.lpt_init(tlinear, a0, order=1)
  state = tfpm.nbody(state, stages, nc)
  tfread = pmutils.cic_paint(tf.zeros_like(tlinear), state[0]).numpy()

  assert_allclose(final_cube, tfread[0], atol=1.2)

def test_rectangular_nody():
  """ Checking end to end nbody on a rectangular grid case
  """
  a0 = 0.1

  pm = ParticleMesh(BoxSize=[bs, bs, 3*bs], Nmesh = [nc, nc, 3*nc], dtype='f4')
  grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
  solver = Solver(pm, Planck15, B=1)
  stages = np.linspace(0.1, 1.0, 10, endpoint=True)

  # Generate initial state with fastpm
  whitec = pm.generate_whitenoise(100, mode='complex', unitary=False)
  lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)
  statelpt = solver.lpt(lineark, grid, a0, order=1)
  finalstate = solver.nbody(statelpt, leapfrog(stages))
  final_cube = pm.paint(finalstate.X)

  # Same thing with flowpm
  tlinear = tf.expand_dims(np.array(lineark.c2r()), 0)
  state = tfpm.lpt_init(tlinear, a0, order=1)
  state = tfpm.nbody(state, stages, [nc, nc, 3*nc])
  tfread = pmutils.cic_paint(tf.zeros_like(tlinear), state[0]).numpy()

  assert_allclose(final_cube, tfread[0], atol=1.2)
