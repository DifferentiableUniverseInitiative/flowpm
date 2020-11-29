# Tests analytic power spectra computations
import tensorflow as tf
import numpy as np
import flowpm.tfbackground as bkgrd
import flowpm.tfpower as power
from nbodykit.cosmology import Planck15
from nbodykit.cosmology.power.linear import LinearPower

from numpy.testing import assert_allclose

def test_eisenstein_hu():
    cosmo={"w0":-1.0,
       "wa":0.0,
       "H0":100,
       "h":Planck15.h,
       "Omega0_b":Planck15.Omega0_b,
       "Omega0_c":Planck15.Omega0_cdm,
       "Omega0_m":Planck15.Omega0_m,
       "Omega0_k":0.0,
       "Omega0_de":Planck15.Omega0_lambda,
       "n_s":Planck15.n_s,
       "sigma8":Planck15.sigma8}

    # Test array of scales
    k = np.logspace(-3, 1, 512)

    # Computing matter power spectrum
    pk_nbodykit = LinearPower(Planck15,0., transfer='EisensteinHu')(k)
    pk_flowpm = power.linear_matter_power(cosmo, k, 1.)

    assert_allclose(pk_nbodykit, pk_flowpm, rtol=1.2e-2)
