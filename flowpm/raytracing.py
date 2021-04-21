#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:29:19 2021

@author: Denise Lanzieri
"""
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import flowpm
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


def convergenceBorn(cosmo,
                    lensplanes,
                    dx,
                    dz,
                    coords,
                    z_source,
                    name="convergenceBorn"):
  """
  Compute the Born–approximated convergence

  Args:
    cosmo: `Cosmology`, cosmology object.
    lensplanes: list of tuples (r, a, lens_plane), lens planes to use 
    dx: float, transverse pixel resolution of the lensplanes [Mpc/h]
    dz: float, width of the lensplanes [Mpc/h]
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.

  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """
  with tf.name_scope(name):
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)

    # Compute constant prefactor:
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    # Compute comoving distance of source galaxies
    r_s = flowpm.background.rad_comoving_distance(cosmo, 1 / (1 + z_source))

    convergence = 0
    for r, a, p in lensplanes:
      density_normalization = dz * r / a
      p = (p - tf.reduce_mean(p, axis=[1, 2], keepdims=True)
           ) * constant_factor * density_normalization
      c = coords * r / dx
      c = tf.expand_dims(c, axis=0) - 0.5

      # Applying periodic conditions on lensplane
      shape = tf.shape(p)
      c = tf.math.mod(c, tf.cast(shape[1], tf.float32))

      im = tfa.image.interpolate_bilinear(tf.expand_dims(p, -1),
                                          c,
                                          indexing='xy')

      convergence += im * tf.reshape(tf.clip_by_value(1. - (r / r_s), 0, 1000),
                                     [1, 1, -1])

    return convergence
