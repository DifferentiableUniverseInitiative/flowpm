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
import flowpm.constants as constants


def density_plane(state,
                  nc,
                  center,
                  width,
                  plane_resolution,
                  name='density_plane'):
  """ Extract a slice from the input state vector and
  project it as a density plane.

  Args:
    state: `Tensor`, input state tensor.
    nc: int or list of int, size of simulation volume
    center: float, center of the slice along the z axis in voxel coordinates
    width: float, width of the slice in voxel coordinates
    plane_size: int, number of pixels of the density plane. 
    name: `string`, name of the operation.

  Returns:
    `Tensor` of shape [batch_size, plane_size, plane_size], of projected density plane.
  """
  with tf.name_scope(name):
    state = tf.convert_to_tensor(state, name="state")
    if isinstance(nc, int):
      nc = [nc, nc, nc]
    nx, ny, nz = nc
    pos = state[0]

    shape = tf.shape(pos)
    batch_size = shape[0]

    xy = pos[..., :2]
    d = pos[..., 2]

    # Apply 2d periodic conditions
    xy = tf.math.mod(xy, nx)

    # Rescaling positions to target grid
    xy = xy / nx * plane_resolution

    # Selecting only particles that fall inside the volume of interest
    mask = (d > (center - width / 2)) & (d <= (center + width / 2))

    # Painting density plane
    density_plane = tf.zeros([batch_size, plane_resolution, plane_resolution])
    density_plane = flowpm.utils.cic_paint_2d(density_plane, xy, mask=mask)

    # Apply density normalization
    density_plane = density_plane / ((nx / plane_resolution) *
                                     (ny / plane_resolution) * (width))

    return density_plane


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
