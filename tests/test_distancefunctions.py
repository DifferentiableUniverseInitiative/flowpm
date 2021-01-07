#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:24:41 2020

@author: Denise
"""

import numpy as np
from flowpm.tfbackground import dchioverda, rad_comoving_distance,a_of_chi as a_of_chi_tf, transverse_comoving_distance as trans_comoving_distance,angular_diameter_distance as ang_diameter_distance
from numpy.testing import assert_allclose

from nbodykit.cosmology import Planck15 as cosmo
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

    
    
def test_radial_comoving_distance():
    """ This function tests the function computing the radial comoving distance.
    """
    a = np.logspace(-2, 0.0)
    
    z=1/a-1
    
    radial =rad_comoving_distance(cosmo1,a)
    
    radial_astr=cosmo.comoving_distance(z)
    
    assert_allclose(radial,radial_astr,rtol=1e-2)
    
    
def test_transverse_comoving_distance():
      """This function test the function computing the Transverse comoving distance in [Mpc/h] for a given scale factor
      """
      a = np.logspace(-2, 0.0)
      
      z=1/a-1

      trans_tf=trans_comoving_distance(cosmo1,a)

      trans_astr=cosmo.comoving_transverse_distance(z)

      assert_allclose(trans_tf, trans_astr, rtol=1e-2)


def test_angular_diameter_distance():
    """This function test the function computing the  Angular diameter distance in [Mpc/h] for a given scale factor
      """
 
    a=np.logspace(-2,0.0)
    
    z=1/a-1
    
    angular_diameter_distance_astr=cosmo.angular_diameter_distance(z)
    
    angular_diameter_distance_tf=ang_diameter_distance(cosmo1,a)
    
    assert_allclose(angular_diameter_distance_tf, angular_diameter_distance_astr,rtol=1e-2)
 
# =============================================================================
# Here we use the nbodykit function that compute comoving_distance as a function of a/z to  
# build a new a-of-chi function by interpolation using a scipy interpolation function. 
# Then we compare thiss function with our a-of-chi function.
# =============================================================================
from scipy import interpolate

def a_of_chi(z):
    r"""Computes the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation.

    Parameters:
    -----------
    cosmo: Cosmology
      Cosmological parameters

    chi: array-like
      radial comoving distance to query.

    Returns:
    --------
    a : array-like
      Scale factors corresponding to requested distances
    """
    a=1/(1+z)
    cache_chi=cosmo.comoving_distance(z)
    return interpolate.interp1d(cache_chi, a)


def test_a_of_chi(): 
    """This function test the function computing the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation
      """ 
      
    a = np.logspace(-2, 0.0)
    
    z=1/a-1
    
    chi = np.geomspace(500, 8000, 50)
      
    aofchi_tf=a_of_chi_tf(cosmo1,chi)
    
    f=a_of_chi(z)
      
    aofchi_astr=f(chi)
      
    assert_allclose(aofchi_tf,aofchi_astr,rtol=1e-2)

# from  jax_cosmo.background import a_of_chi as a_of_chi_jax
# import jax_cosmo as jc
# cosmo_jax= jc.Planck15()


# def test_a_of_chi2(): 
#     """This function test the function computing the scale factor for corresponding (array) of radial comoving
#     distance by reverse linear interpolation
#       """ 
     
#     #Comparing a-of-chi jax with a of chi scipy
#     a = np.logspace(-2, 0.0)
    
#     z=1/a-1
    
#     chi = np.linspace(500, 8000, 50)
      
#     aofchi_jax=a_of_chi_jax(cosmo_jax,chi)
    
#     f=a_of_chi(z)
      
#     aofchi_astr=f(chi)
      
#     assert_allclose(aofchi_jax,aofchi_astr,rtol=1e-2)


# def test_a_of_chi3(): 
#     """This function test the function computing the scale factor for corresponding (array) of radial comoving
#     distance by reverse linear interpolation
#       """ 
#       #Comparing a-of-chi jax with a of chi tensorflow
    
#     chi = np.linspace(500, 8000, 50)
      
#     aofchi_jax=a_of_chi_jax(cosmo_jax,chi)
    
#     aofchi_tf=a_of_chi_tf(cosmo1,chi)
      
#     assert_allclose(aofchi_jax,aofchi_tf,rtol=1e-2)



