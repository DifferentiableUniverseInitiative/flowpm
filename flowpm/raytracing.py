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
import flowpm.constants as constants


def lightcone(
    cosmo,
    state,
    stages,
    nc,
    plane_resolution,  # in arcmin
    plane_size,  # in pixels
    pm_nc_factor=1,
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

    # Compute the width of the lens planes based on number of time steps
    w = nc[2] // (nstages - 1)
    nx = nc[0]
    nz = nc[2]
    lps = []
    lps_a = []

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

      # This is the transverse comoving distance inside the box
      xy = pos[:, :, :2] - nx / 2

      # Compute density plane in sky coordinates around the center of the lightcone
      # TODO: Confirm conversion from comoving distances to angular size! I thought
      # we should be using the angular diameter distance, but as far as I can see
      # everyone uses the transverse comoving distance, and I don't understand exactly why
      lens_plane = tf.zeros([batch_size, plane_size, plane_size])

      # Convert coordinates to angular coords, and then into plane coords
      xy = (xy / tf.expand_dims(d, -1)) / np.pi * 180 * 60 / plane_resolution
      xy = xy + plane_size / 2

      # Selecting only the particles contributing to the lens plane
      mask = tf.where((d > (nz - (i + 1) * w)) & (d <= (nz - i * w)), 1., 0.)
      # And falling inside the plane, NOTE: This is only necessary on CPU, on GPU
      # cic paint 2d can be made to work with non periodic conditions.
      mask = mask * tf.where(
          (xy[..., 0] > 0) & (xy[..., 0] < plane_size), 1., 0.)
      mask = mask * tf.where(
          (xy[..., 1] > 0) & (xy[..., 1] < plane_size), 1., 0.)
      # Compute lens planes by projecting particles
      lens_plane = flowpm.utils.cic_paint_2d(lens_plane, xy + plane_size / 2,
                                             mask)
      lps.append(lens_plane)
      lps_a.append(ah)

      # Here we could trim the state vector for particles originally beyond the current lens plane
      # This way the simulation becomes smaller as it runs and we save resources
      state = tf.reshape(state, [3, batch_size, nc[0], nc[1], -1, 3])
      state = state[:, :, :, :, :(
          nz - i * w -
          w // 2), :]  # We keep w/2 to be safe, so we allow particle to travel
      # A max distance of width/2
      # redefine shape of state
      nc = state.get_shape()[2:5]
      state = tf.reshape(state, [3, batch_size, -1, 3])
      # So this seems to work, but we should be a tiny bit careful because we break periodicity in the z
      # direction at z=0.... probably not a big deal but still gotta check what that does.

      # Force
      state = force(cosmo, state, nc, pm_nc_factor=pm_nc_factor)
      f = a1

      # Kick again
      state = kick(cosmo, state, p, f, a1)
      p = a1

    return state, lps_a, lps


def wlen(ds, a, nc, Boxsize, plane_size, field, cosmo):
  """
    Returns the correctly weighted lensing efficiency kernel

    Parameters:
    -----------
    ds: float
        comoving source distance

    a : array_like or tf.TensorArray
        Scale factor

    nc : list
        Size of the cube, number of cells

    Boxsize: list
        Physical size of the cube

    plane_size : int
       Number of pixels for x and  y

    field: int or float
        transveres degres of the field

    Returns
    -------
    w : Scalar float Tensor
        Weighted lensing efficiency kernel

    """
  d = rad_comoving_distance(cosmo, a)

  # 2D mesh area in rad^2 per pixel
  A = (field * np.pi / 180 / plane_size)**2

  # mean 3D particle density
  nbar = np.prod(nc) / np.prod(Boxsize)

  # particles/Volume*angular pixel area* distance^2 -> 1/L units
  columndens = (A * nbar) * (d**2)
  w = ((ds - d) * (d / ds)) / (columndens)
  w = w / a
  return w


def Born(lps_a, lps, ds, nc, Boxsize, plane_size, field, cosmo):
  """
    Compute the Born–approximated convergence

    Parameters:
    -----------
    lps_a : tf.TensorArray
        Scale factor of lens planes

    lps : tf.TensorArray
        density field of each lens plane

    ds: float
        comoving source distance

    Returns
    -------
    k_map : tf.TensorArray
        Born–approximated convergence

    """
  k_map = 0

  # Compute constant prefactor:
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2

  for i in range(len(lps_a)):
    k_map += constant_factor * lps[i][0] * wlen(ds, lps_a[i], nc, Boxsize,
                                                plane_size, field, cosmo)
  return k_map
