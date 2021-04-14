#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:19:45 2021

@author: Denise Lanzieri
"""

import tensorflow as tf
import numpy as np
import flowpm
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from DifferentiableHOS.pk import pk as pkl
from numpy.testing import assert_allclose


#%%
z_source=1.
field=5.
box_size=200.
nc=16
Omega_c= 0.2589
sigma8= 0.8159
nsteps=2

#%%
def compute_initial_cond(Omega_c):
    cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c)
    k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
    pk = tfpower.linear_matter_power(cosmology, k)
    pk_fun = lambda x: tf.cast(tf.reshape(interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.complex64)
    initial_conditions = flowpm.linear_field([nc, nc, nc],
                                           [box_size, box_size,
                                           box_size],
                                           pk_fun,
                                           batch_size=1)
    return initial_conditions 



@tf.function
def compute_powerspectrum(Omega_c):
    cosmology = flowpm.cosmology.Planck15(Omega_c=Omega_c)
    stages = np.linspace(0.1, 1., nsteps, endpoint=True)
    state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)
    final_state = flowpm.nbody(cosmology, state, stages, [nc, nc,  nc])         
    final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])
    final_field=tf.reshape(final_field, [nc, nc, nc])
    k, power_spectrum = pkl(final_field,shape=final_field.shape,boxsize=np.array([box_size, box_size,
                                               box_size]),kmin=0.1,dk=2*np.pi/box_size)
    return  power_spectrum


@tf.function
def Flow_jac(Omega_c):
    params = tf.stack([Omega_c])
    with tf.GradientTape() as tape:
        tape.watch(params)
        cosmology = flowpm.cosmology.Planck15(Omega_c=params[0])
        stages = np.linspace(0.1, 1., nsteps, endpoint=True)
        state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

        final_state = flowpm.nbody(cosmology, state, stages, [nc, nc,  nc])         
        final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])
        final_field=tf.reshape(final_field, [nc, nc, nc])
        k, power_spectrum = pkl(final_field,shape=final_field.shape,boxsize=np.array([box_size, box_size,
                                               box_size]),kmin=0.1,dk=2*np.pi/box_size)
    return tape.jacobian(power_spectrum, params,experimental_use_pfor=False)

#%%
initial_conditions=compute_initial_cond(0.2589) 

def test_Nbody_jacobian():
    theoretical, numerical_jac=tf.test.compute_gradient( compute_powerspectrum, [Omega_c], delta=0.01)
    FlowPM_jac= Flow_jac(Omega_c)
    assert_allclose(numerical_jac[0],FlowPM_jac, rtol=1e-1)
    
    
    
 