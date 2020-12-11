#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:24:41 2020

@author: dl264294
"""

import numpy as np
from flowpm.tfbackground import dchioverda, rad_comoving_distance,a_of_chi as a_of_chi_tf, transverse_comoving_distance as trans_comoving_distance,angular_diameter_distance as ang_diameter_distance
from numpy.testing import assert_allclose
from  jax_cosmo.background import dchioverda as dchio
from  jax_cosmo.background import radial_comoving_distance, a_of_chi, transverse_comoving_distance,angular_diameter_distance
import jax_cosmo as jc
cosmo = jc.Planck15()


cosmo1={"w0":-1.0,
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


def test_dchioverda():
    """ This function tests the scale factor dependence of the
    Hubble parameter.
    """
    
    a = np.logspace(-3, 0.0)

    dchioverda_ref = dchioverda(cosmo1,a)
    
    dchioverda_back = dchio(cosmo, a)

    assert_allclose(dchioverda_ref, dchioverda_back, rtol=1e-4)
    
    
def test_radial_comoving_distance():
    """ This function tests the function computing the radial comoving distance.
    """
    a = np.logspace(-3, 0.0)
 
    radial =rad_comoving_distance(cosmo1,a)
    
    radial_jax=radial_comoving_distance(cosmo,a)
    
    assert_allclose(radial,radial_jax,rtol=1e-3)
    
    

def test_a_of_chi(): 
   """This function test the function computing the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation
      """ 
   chi = np.linspace(1000,100)
      
   aofchi=a_of_chi_tf(cosmo1,chi)
      
   aofchi_jax=a_of_chi(cosmo,chi)
      
   assert_allclose(aofchi,aofchi_jax,rtol=1e-3)
  
    
def test_transverse_comoving_distance():
      """This function test the function computing the Transverse comoving distance in [Mpc/h] for a given scale factor
      """
      a = np.logspace(-3, 0.0)

      trans_tf=trans_comoving_distance(cosmo1,a)

      trans_jax=transverse_comoving_distance(cosmo,a)

      assert_allclose(trans_tf, trans_jax, rtol=1e-3)


def test_angular_diameter_distance():
    """This function test the function computing the  Angular diameter distance in [Mpc/h] for a given scale factor
      """
 
    a=np.logspace(-3,0.0)
    
    angular_diameter_distance_jax=angular_diameter_distance(cosmo,a)
    
    angular_diameter_distance_tf=ang_diameter_distance(cosmo1,a)
    
    assert_allclose(angular_diameter_distance_tf, angular_diameter_distance_jax,rtol=1e-3)