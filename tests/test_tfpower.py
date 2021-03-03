# Tests analytic power spectra computations
import tensorflow as tf
import numpy as np
import flowpm
import flowpm.tfpower as power
from nbodykit.cosmology import Planck15
from nbodykit.cosmology.power.linear import LinearPower

from numpy.testing import assert_allclose


def test_eisenstein_hu():
  cosmo = flowpm.cosmology.Cosmology(Omega_c=Planck15.Omega0_cdm,
                                     Omega_b=Planck15.Omega0_b,
                                     Omega0_k=0.0,
                                     h=Planck15.h,
                                     n_s=Planck15.n_s,
                                     sigma8=Planck15.sigma8,
                                     w0=-1.,
                                     wa=0.0)
  # Test array of scales
  k = np.logspace(-3, 1, 512)

  # Computing matter power spectrum
  pk_nbodykit = LinearPower(Planck15, 0., transfer='EisensteinHu')(k)
  pk_flowpm = power.linear_matter_power(cosmo, k, 1.)

  assert_allclose(pk_nbodykit, pk_flowpm, rtol=1.2e-2)
