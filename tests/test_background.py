import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose

import flowpm.tfbackground as tfbackground
from flowpm.background import MatterDominated

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

def test_E():
  """ This function tests the scale factor dependence of the
  Hubble parameter.
  """
  M_d=MatterDominated(Omega0_m=0.3075)
  a = np.logspace(-3, 0)
  # Computing reference E value with old code
  E_ref = M_d.E(a)
  # Computing new E function with tensorflow
  E = tfbackground.E(cosmo, a)

  assert_allclose(E_ref, E, rtol=1e-4)
