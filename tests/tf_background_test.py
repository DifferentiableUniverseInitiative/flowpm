import tensorflow as tf
import numpy as np
from flowpm.tfbackground import E,dEa,Omega_m_a,odesolve_func
from numpy.testing import assert_allclose
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
    E_back = E(cosmo, a)

    assert_allclose(E_ref, E_back, rtol=1e-4)


def test_Eprime():
  """ Testing Derivative of the scale factor dependent factor E(a)
  """
  M_d=MatterDominated(Omega0_m=0.3075)
  a = np.logspace(-3, 0)
  # Computing reference E' value with old code
  E_prim_back=M_d.efunc_prime(a)
  # Computing new E' function with tensorflow
  E_n = dEa(cosmo, a)
  
  assert_allclose(E_prim_back, E_n, rtol=1e-4)


def test_Omega_m():
  """ Testing Matter density at scale factor `a`
  """
  M_d=MatterDominated(Omega0_m=0.3075)
  a = np.logspace(-3, 0)
  # Computing reference Omega_m value with old code
  Omega_back=M_d.Om(a)
  # Computing new Omega_m' function with tensorflow
  Omega_m_n = Omega_m_a(cosmo, a)

  assert_allclose(Omega_back, Omega_m_n, rtol=1e-4)




def test_growth_1order():
    """ Testing linear growth factor D_1(a) 
    """
    
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0)
    log10_amin=-2
    steps=128
    atab =np.logspace(log10_amin, 0.0, steps)
    y0=tf.constant([[atab[0], -3./7 * 0.01**2], [1.0, -6. / 7 *0.01]],dtype=tf.float32)
    gback = M_d.D1(a)
    results_func=odesolve_func(a,y0)
    gtfback =results_func.states[:,0,0]/results_func.states[-1,0,0]

    assert_allclose(gback, gtfback, rtol=1e-2)


def test_growth_2order():
    """ Testing linear growth factor D_2(a) 
    """
    
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0)
    log10_amin=-2
    steps=128
    atab =np.logspace(log10_amin, 0.0, steps)
    y0=tf.constant([[atab[0], -3./7 * 0.01**2], [1.0, -6. / 7 *0.01]],dtype=tf.float32)
    g2back = M_d.D2(a)
    results_func=odesolve_func(a,y0)
    g2tfback =results_func.states[:,0,1]/results_func.states[-1,0,1]

    assert_allclose(g2back, g2tfback, rtol=1e-2)
# =============================================================================
