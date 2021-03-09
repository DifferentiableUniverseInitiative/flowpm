import tensorflow as tf
import tempfile
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from numpy.testing import assert_allclose
import flowpm
from flowpm.io import save_state
from flowpm.tfpower import linear_matter_power
import bigfile

np.random.seed(0)


def test_save_state():
  """
  Tests the BigFile saving function
  """
  klin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)
  a0 = 0.1
  nc = [16, 16, 16]
  boxsize = [100., 100., 100.]
  cosmo = flowpm.cosmology.Planck15()

  initial_conditions = flowpm.linear_field(
      nc,  # size of the cube
      boxsize,  # Physical size of the cube
      ipklin,  # Initial powerspectrum
      batch_size=2)

  # Sample particles
  state = flowpm.lpt_init(cosmo, initial_conditions, a0)

  with tempfile.TemporaryDirectory() as tmpdirname:
    filename = tmpdirname + '/testsave'
    save_state(cosmo, state, a0, nc, boxsize, filename)

    # Now try to reload the information using BigFile
    bf = bigfile.BigFile(filename)

    # Testing recovery of header
    header = bf['Header']
    assert_allclose(np.array(header.attrs['NC']), np.array(nc))
    assert_allclose(np.array(header.attrs['BoxSize']), np.array(boxsize))
    assert_allclose(np.array(header.attrs['OmegaCDM']),
                    np.array(cosmo.Omega_c))
    assert_allclose(np.array(header.attrs['OmegaB']), np.array(cosmo.Omega_b))
    assert_allclose(np.array(header.attrs['OmegaK']), np.array(cosmo.Omega_k))
    assert_allclose(np.array(header.attrs['h']), np.array(cosmo.h))
    assert_allclose(np.array(header.attrs['Sigma8']), np.array(cosmo.sigma8))
    assert_allclose(np.array(header.attrs['w0']), np.array(cosmo.w0))
    assert_allclose(np.array(header.attrs['wa']), np.array(cosmo.wa))
    assert_allclose(np.array(header.attrs['Time']), np.array(a0))

    # Testing recovery of data
    pos = bf['1/Position']
    assert_allclose(pos[:], state[0, 1].numpy() / nc[0] * boxsize[0])
    vel = bf['1/Velocity']
    assert_allclose(vel[:], state[1, 1].numpy() / nc[0] * boxsize[0])

    # Closing file
    bf.close()
