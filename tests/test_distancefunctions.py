#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:24:41 2020

@author: Denise
"""

import numpy as np
import flowpm
from flowpm.tfbackground import dchioverda, rad_comoving_distance, a_of_chi as a_of_chi_tf, transverse_comoving_distance as trans_comoving_distance, angular_diameter_distance as ang_diameter_distance
from numpy.testing import assert_allclose
from scipy import interpolate
from nbodykit.cosmology import Planck15 as cosmo


def test_radial_comoving_distance():
  """ This function tests the function computing the radial comoving distance.
  """
  cosmo_tf = flowpm.cosmology.Planck15()
  a = np.logspace(-2, 0.0)

  z = 1 / a - 1

  radial = rad_comoving_distance(cosmo_tf, a)

  radial_astr = cosmo.comoving_distance(z)

  assert_allclose(radial, radial_astr, rtol=1e-2)


def test_transverse_comoving_distance():
  """This function test the function computing the Transverse comoving distance in [Mpc/h] for a given scale factor
  """
  cosmo_tf = flowpm.cosmology.Planck15()
  a = np.logspace(-2, 0.0)

  z = 1 / a - 1

  trans_tf = trans_comoving_distance(cosmo_tf, a)

  trans_astr = cosmo.comoving_transverse_distance(z)

  assert_allclose(trans_tf, trans_astr, rtol=1e-2)


def test_angular_diameter_distance():
  """This function test the function computing the  Angular diameter distance in [Mpc/h] for a given scale factor
  """
  cosmo_tf = flowpm.cosmology.Planck15()

  a = np.logspace(-2, 0.0)

  z = 1 / a - 1

  angular_diameter_distance_astr = cosmo.angular_diameter_distance(z)

  angular_diameter_distance_tf = ang_diameter_distance(cosmo_tf, a)

  assert_allclose(angular_diameter_distance_tf,
                  angular_diameter_distance_astr,
                  rtol=1e-2)


# =============================================================================
# Here we use the nbodykit function that compute comoving_distance as a function of a/z to
# build a new a-of-chi function by interpolation using a scipy interpolation function.
# Then we compare thiss function with our a-of-chi function.
# =============================================================================


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
  a = 1 / (1 + z)
  cache_chi = cosmo.comoving_distance(z)
  return interpolate.interp1d(cache_chi, a, kind='cubic')


def test_a_of_chi():
  """This function test the function computing the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation
  """
  cosmo_tf = flowpm.cosmology.Cosmology(Omega_c=cosmo.Omega0_cdm,
                                     Omega_b=cosmo.Omega0_b,
                                     Omega_k=0.0,
                                     h=cosmo.h,
                                     n_s=cosmo.n_s,
                                     sigma8=cosmo.sigma8,
                                     w0=-1.,
                                     wa=0.0)

  a = np.logspace(-2, 0.0, 512)

  z = 1 / a - 1

  chi = np.geomspace(100, 6000, 50)

  aofchi_tf = a_of_chi_tf(cosmo_tf, chi)

  f = a_of_chi(z)

  aofchi_astr = f(chi)

  assert_allclose(aofchi_tf, aofchi_astr, rtol=1e-2)
