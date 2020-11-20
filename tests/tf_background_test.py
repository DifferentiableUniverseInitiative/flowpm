import tensorflow as tf
import numpy as np
from flowpm.tfbackground import dEa,Omega_m_a, F1, E, F2, Gf,Gf2, gf,gf2,D1_norm,D2_norm,D1f_norm,D2f_norm
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
    a =np.logspace(-2, 0.0, 128)
    gback = M_d.D1(a)
    gtfback =D1_norm(a)

    assert_allclose(gback, gtfback, rtol=1e-2)


def test_growth_2order():
    """ Testing linear growth factor D_2(a) 
    """
    
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0.0,128)
    g2back = M_d.D2(a)
    g2tfback =D2_norm(a)

    assert_allclose(g2back, g2tfback, rtol=1e-2)
    
def test_D1_fnorm():
    """ Testing  D'_1(a) 
    """
    
    M_d=MatterDominated(Omega0_m=0.3075)
    a =np.logspace(-2, 0.0, 128)
    gback = M_d.gp(a)
    gtfback =D1f_norm(a)

    assert_allclose(gback, gtfback, rtol=1e-2)


def test_D2_fnorm():
    """ Testing  D'_2(a)
    """
    
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0.0,128)
    g2back = M_d.gp2(a)
    g2tfback =D2f_norm(a)

    assert_allclose(g2back, g2tfback, rtol=1e-2)    
    
    
# =============================================================================
def testf1():
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0,128)
    f1_back=M_d.f1(a)
    f1_tf=F1(a)
    
    assert_allclose(f1_back, f1_tf, rtol=1e-2)
    
    
    
def testf2():
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0,128)
    f2_back=M_d.f2(a)
    f2_tf=F2(a)
    
    assert_allclose(f2_back, f2_tf, rtol=1e-2)
    
    
def testGf():
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0,128)
    Gf_back=M_d.Gf(a)
    Gf_tf=Gf(a)
    
    assert_allclose(Gf_back, Gf_tf, rtol=1e-2)
    
def testGf2():
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0,128)
    Gf2_back=M_d.Gf2(a)
    Gf2_tf=Gf2(a)
    
    assert_allclose(Gf2_back, Gf2_tf, rtol=1e-2)
    
    
def testgf():
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0,128)
    gf_back=M_d.gf(a)
    gf_tf=gf(a)
    
    assert_allclose(gf_back, gf_tf, rtol=1e-2)
    
def testgf2():
    M_d=MatterDominated(Omega0_m=0.3075)
    a = np.logspace(-2, 0,128)
    gf2_back=M_d.gf2(a)
    gf2_tf=gf2(a)
    
    assert_allclose(gf2_back, gf2_tf, rtol=1e-2)
    