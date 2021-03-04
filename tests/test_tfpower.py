# Tests analytic power spectra computations
import tensorflow as tf
import numpy as np
import flowpm
import flowpm.tfpower as power
from nbodykit.cosmology import Cosmology
from nbodykit.cosmology.power.linear import LinearPower
from astropy.cosmology import Planck15
import astropy.units as u
from numpy.testing import assert_allclose

# Create a simple Planck15 cosmology without neutrinos, and makes sure sigma8
# is matched
ref_cosmo = Cosmology.from_astropy(Planck15.clone(m_nu=0 * u.eV))
ref_cosmo = ref_cosmo.match(sigma8=flowpm.cosmology.Planck15().sigma8.numpy())


def test_eisenstein_hu():
  cosmo = flowpm.cosmology.Planck15()

  # Test array of scales
  k = np.logspace(-3, 1, 512)

  # Computing matter power spectrum
  pk_nbodykit = LinearPower(ref_cosmo, 0., transfer='EisensteinHu')(k)
  pk_flowpm = power.linear_matter_power(cosmo, k, 1.)

  assert_allclose(pk_nbodykit, pk_flowpm, rtol=1.2e-2)
