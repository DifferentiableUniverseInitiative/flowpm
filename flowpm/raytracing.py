#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:29:19 2021

@author: Denise Lanzieri
"""
import numpy as np
import tensorflow as tf
import flowpm
from flowpm.tfbackground import rad_comoving_distance
from flowpm.tfpm import kick, drift, force


def lightcone(
    cosmo,
    state,
    stages,
    nc,
    plane_resolution,  # in arcmin
    plane_size,  # in pixels
    pm_nc_factor=1,
    save_snapshots=False,
    name="NBody"):
  """
  Integrate the evolution of the state across the givent stages

  Parameters:
  -----------
  cosmo: cosmology
    Cosmological parameter object

  state: tensor (3, batch_size, npart, 3)
    Input state

  stages: array
    Array of scale factors, also define slices in the volume

  nc: int
    Number of cells

  pm_nc_factor: int
    Upsampling factor for computing

  Returns
  -------
  state: tensor (3, batch_size, npart, 3)
    Integrated state to final condition
  """
  with tf.name_scope(name):
    state = tf.convert_to_tensor(state, name="state")

    shape = state.get_shape()
    batch_size = shape[1]

    nstages, = stages.shape

    # Unrolling leapfrog integration to make tf Autograph happy
    if nstages == 0:
      return state

    ai = stages[0]

    # first force calculation for jump starting
    state = force(cosmo, state, nc, pm_nc_factor=pm_nc_factor)
    nx = nc[0]
    nz = nc[2]
    lps = []
    lps_a = []
    snapshots = []
    w = 64

    x, p, f = ai, ai, ai
    # Loop through the stages
    for i in range(nstages - 1):
      a0 = stages[i]
      a1 = stages[i + 1]
      ah = (a0 * a1)**0.5

      # Kick step
      state = kick(cosmo, state, p, f, ah)
      p = ah

      # Drift step
      state = drift(cosmo, state, x, p, a1)
      x = a1

      # Access the positions of the particles
      pos = state[0]
      d = pos[:, :, 2]

      # Compute density plane in sky coordinates around the center of the lightcone
      # TODO: Confirm conversion from comoving distances to angular size! I thought
      # we should be using the angular diameter distance, but as far as I can see
      # everyone uses the transverse comoving distance, and I don't understand exactly why
      lens_plane = tf.zeros([batch_size, plane_size, plane_size])

      # Selecting only the particles contributing to the lens plane
      s = tf.gather(state,
                    tf.where((d[0] > (nz - (i + 1) * w))
                             & (d[0] <= (nz - i * w)))[:, 0],
                    axis=2)
      # NOTES!!! IT WILL ONLY WORK FOR BATCH 1

      # This is the transverse comoving distance inside the box
      xy = s[0, :, :, :2]
      d = s[0, :, :, 2]
      # Apply periodic conditions
      xy = tf.math.mod(xy, nx)

      # Convert coordinates to angular coords, and then into plane coords
      # xy = (xy / tf.expand_dims(d, -1)) / np.pi * 180 * 60 / plane_resolution

      xy = xy / nx * plane_size

      # mask = tf.where((d > (nz - (i + 1) * w)) & (d <= (nz - i * w)), 1., 0.)

      # And falling inside the plane, NOTE: This is only necessary on CPU, on GPU
      # cic paint 2d can be made to work with non periodic conditions.
      mask = tf.where((xy[..., 0] > 0) & (xy[..., 0] < plane_size), 1., 0.)
      mask = mask * tf.where(
          (xy[..., 1] > 0) & (xy[..., 1] < plane_size), 1., 0.)
      # Compute lens planes by projecting particles
      lens_plane = flowpm.utils.cic_paint_2d(lens_plane, xy, mask)
      # Subtracting mean: Is this the right thing to do???
      #lens_plane = lens_plane - tf.reduce_mean(lens_plane, axis=[1,2], keepdims=True)
      #
      lps.append(lens_plane)
      lps_a.append(a1)

      # Here we could trim the state vector for particles originally beyond the current lens plane
      # This way the simulation becomes smaller as it runs and we save resources
      #s = tf.reshape(state, [3, batch_size, nc[0], nc[1], -1, 3])
      #s = s[:, :, :, :, (nz - (i + 1) * w):(nz - i * w), :]  #- tf.constant([0.,0.,(nz - (i + 1) * w)])
      #print(i, (nz - (i + 1) * w), (nz - i * w))
      snapshots.append(s)  #tf.reshape(s, [3, batch_size, -1, 3]))

      # Force
      state = force(cosmo, state, nc, pm_nc_factor=pm_nc_factor)
      f = a1

      # Kick again
      state = kick(cosmo, state, p, f, a1)
      p = a1
    #return state, lps_a, lps, snapshots
    if save_snapshots:
      return state, lps_a, lps, snapshots
    else:
      return state, lps_a, lps


def convergenceBorn(interp_im, rl, initial_positions, z, plane_size):
  """
    Compute the Born–approximated convergence
    
    Parameters:
    -----------
    interp_im : tf.TensorArray
        density field of each lens plane projected on the sky
        
    rl : tf.TensorArray
        Comoving distances of lens planes
    
    initial_positions : Astropy.units.quantity
        Initial angular positions of the light ray bucket, according to the observer
    
    z: float
        source redshift
        
    plane_size : int
       Number of pixels for x and  y
    
    Returns
    -------
    k_map : tf.TensorArray
        Born–approximated convergence
    
    """
  a_source = 1. / (1. + z)
  cosmo = flowpm.cosmology.Planck15()
  current_convergence = np.zeros(initial_positions.shape[1:])
  distance = rl
  last_lens = len(interp_im)
  for k in range(last_lens):
    plans = interp_im[k]
    density = plans[0, :, 0].numpy().reshape([plane_size, plane_size]).T
    current_convergence += density * (
        1. - (distance[k] / rad_comoving_distance(cosmo, a_source)))
  return current_convergence
