# Tests analytic power spectra computations
import tensorflow as tf
import numpy as np
import flowpm.tfbackground as bkgrd
import flowpm.tfpower as power
import pyccl as ccl

from numpy.testing import assert_allclose

def test_eisenstein_hu():
    # We first define equivalent cosmologies
    cosmo_ccl = ccl.Cosmology(
        Omega_c=0.2589,
        Omega_b=0.0486,
        h=0.6774,
        sigma8=0.8159,
        n_s=0.9667,
        Neff=0,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="linear",
    )
    cosmo={"w0":-1.0,
       "wa":0.0,
       "H0":100,
       "h":0.6774,
       "Omega0_b":0.04860,
       "Omega0_c":0.2589,
       "Omega0_m":0.3075,
       "Omega0_k":0.0,
       "Omega0_de":0.6925,
       "n_s":0.9667,
       "sigma8":0.8159}

    # Test array of scales
    k = np.logspace(-4, 2, 512)

    # Computing matter power spectrum
    pk_ccl = ccl.linear_matter_power(cosmo_ccl, k, 0.1)
    pk_flowpm = power.linear_matter_power(cosmo, k / cosmo['h'], 0.1)/ cosmo['h'] ** 3

    assert_allclose(pk_ccl, pk_flowpm, rtol=0.5e-2)
