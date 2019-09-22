"""Module storing a few tensorflow function to implement FastPM """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def cic_paint(mesh, part, weight=None, name=None):
  """
  Paints particules on a 3D mesh.

  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc, nc)
    Input 3D mesh tensor

  part: tensor (batch_size, npart, 3)
    List of 3D particle coordinates, assumed to be in mesh units if
    boxsize is None

  weight: tensor (batch_size, npart)
    List of weights  for each particle
  """
  with tf.name_scope(name, "CiCPaint", [mesh, part, weight]):
    shape = tf.shape(mesh)
    batch_size, nc = shape[0], shape[1]

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(tf.constant([[[0, 0, 0], [1., 0, 0],[0., 1, 0],
                                              [0., 0, 1],[1., 1, 0],[1., 0, 1],
                                              [0., 1, 1],[1., 1, 1]]]), 0)

    neighboor_coords = floor + connection
    kernel = 1. - tf.abs(part - neighboor_coords)
    # Replacing the reduce_prod op by manual multiplication
    # TODO: figure out why reduce_prod was crashing the Hessian computation
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    if weight is not None: kernel = tf.multiply(tf.expand_dims(weight, axis=-1) , kernel)

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)
    neighboor_coords = tf.math.mod(neighboor_coords , nc)

    # Adding batch dimension to the neighboor coordinates
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, [1] + list(neighboor_coords.get_shape()[1:-1]) + [1])
    neighboor_coords = tf.concat([b, neighboor_coords], axis=-1)

    update = tf.scatter_nd(tf.reshape(neighboor_coords, (-1, 8,4)),
                           tf.reshape(kernel, (-1, 8)),
                           [batch_size, nc, nc, nc])
    mesh = mesh + update
    return mesh

def cic_readout(mesh, part, name=None):
  """
  Reads out particles from mesh.

  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc, nc)
    Input 3D mesh tensor

  part: tensor (batch_size, npart, 3)
    List of 3D particle coordinates, assumed to be in mesh units if
    boxsize is None

  Return:
  -------
  value: tensor (batch_size, npart)
    Value of the field sampled at the particle locations
  """
  with tf.name_scope(name, "CiCReadout", [mesh, part]):
    shape = tf.shape(mesh)
    batch_size, nc = shape[0], shape[1]

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(tf.constant([[[0, 0, 0], [1., 0, 0],[0., 1, 0],
                                              [0., 0, 1],[1., 1, 0],[1., 0, 1],
                                              [0., 1, 1],[1., 1, 1]]]), 0)

    neighboor_coords = tf.add(floor, connection)
    kernel = 1. - tf.abs(part - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)
    neighboor_coords = tf.math.mod(neighboor_coords , nc)

    meshvals = tf.gather_nd(mesh, neighboor_coords, batch_dims=1)
    weightedvals = tf.multiply(meshvals, kernel)
    value = tf.reduce_sum(weightedvals, axis=-1)
    return value

def r2c3d(rfield, norm=None, dtype=tf.complex64, name=None):
  """
  Converts a real field to its complex Fourier Transform

  Parameters:
  -----------
  rfield: tensor (batch_size, nc, nc, nc)
    Input 3D real field

  norm: float
    Normalization factor

  dtype: tf.dtype
    Type of output tensor

  Return:
  -------
  cfield: tensor (batch_size, nc, nc, nc)
    Complex field
  """
  with tf.name_scope(name, "R2C3D", [rfield]):
    if norm is None: norm = tf.cast(tf.reduce_prod(rfield.get_shape()[1:]), dtype)
    else: norm = tf.cast(norm, dtype)
    cfield = tf.multiply(tf.signal.fft3d(tf.cast(rfield, dtype)), 1/norm, name=name)
    return cfield

def c2r3d(cfield, norm=None, dtype=tf.float32, name=None):
  """
  Converts a complex Fourier domain field to a real field

  Parameters:
  -----------
  cfield: tensor (batch_size, nc, nc, nc)
    Complex 3D real field

  norm: float
    Normalization factor

  dtype: tf.dtype
    Type of output tensor

  Return:
  -------
  rfield: tensor (batch_size, nc, nc, nc)
    Real valued field
  """
  with tf.name_scope(name, "C2R3D", [cfield]):
    if norm is None: norm = tf.cast(tf.reduce_prod(cfield.get_shape()[1:]), dtype)
    else: norm = tf.cast(norm, dtype)
    rfield = tf.multiply(tf.cast(tf.signal.ifft3d(cfield), dtype), norm, name=name)
    return rfield

def white_noise(nc, batch_size=1, seed=None, type='complex', name=None):
  """
  Samples a 3D cube of white noise of desired size
  """
  with tf.name_scope(name, "WhiteNoise"):
    assert batch_size >= 1
    white = tf.random_normal(shape=(batch_size, nc, nc, nc),
                             mean=0, stddev=nc**1.5, seed=seed)
    if type == 'real': return white
    elif type == 'complex':
        whitec = r2c3d(white, norm=nc**3)
        return whitec
