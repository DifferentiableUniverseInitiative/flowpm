#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:48:34 2021

@author: Denise Lanzieri
"""
import numpy as np
from numpy.testing import assert_allclose
import lenstools as lt
import astropy.units as u
from DifferentiableHOS.lenstools_script import lenstools_raytracer, compute_plans
from flowpm.raytracing import Born
from astropy.cosmology import Planck15
import flowpm
 
#%%
import pickle
# data= pickle.load( open("/Users/dl264294/Desktop/github/DifferentiableHOS/scripts/different_results.pkl", "rb" ))

# lps=data['lps']
# lps_a=data['lps_a']
# ds=data['ds']
#%%
z_source=1.
field=5.
box_size=200
nc=64
plane_size=64
Omega_c= 0.2589
sigma8= 0.8159
nsteps=20
   
cosmo=flowpm.cosmology.Planck15()
#%%

def test_convergence():
    """ This function tests the Born approximation implemented in TensorFlow 
    comparing it with Lenstools
      """
    k_map, lps_a, lps, ds= compute_plans(nc,plane_size,box_size,field,z_source,nsteps,cosmo)
    lt_map= lenstools_raytracer(lps_a,lps,nc,plane_size,box_size,field,z_source,cosmo)
    kmap_flowpm = lt.ConvergenceMap(k_map, 5*u.deg)
    kmap_lt = lt.ConvergenceMap(lt_map, 5*u.deg)
    l_edges = np.arange(200.0,5000.0,200.0)
    l1,Pl1 = kmap_flowpm.powerSpectrum(l_edges)
    l2,Pl2 = kmap_lt.powerSpectrum(l_edges)
    
    assert_allclose(Pl1, Pl2, rtol=1e-3)

