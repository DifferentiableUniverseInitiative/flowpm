"""Module storing a few tensorflow function to implement FastPM """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def cic_paint(mesh, part, weight=None, name="CiCPaint"):
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
  with tf.name_scope(name):
    mesh = tf.convert_to_tensor(mesh, name="mesh")
    part = tf.convert_to_tensor(part, name="part")
    if weight is not None:
      weight = tf.convert_to_tensor(weight, name="weight")

    shape = tf.shape(mesh)
    batch_size, nx, ny, nz = shape[0], shape[1], shape[2], shape[3]
    nc = [nx, ny, nz]

    # Flatten part if it's not already done
    if len(part.shape) > 3:
      part = tf.reshape(part, (batch_size, -1, 3))

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(
        tf.constant([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                      [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]]), 0)

    neighboor_coords = floor + connection
    kernel = 1. - tf.abs(part - neighboor_coords)
    # Replacing the reduce_prod op by manual multiplication
    # TODO: figure out why reduce_prod was crashing the Hessian computation
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    if weight is not None:
      kernel = tf.multiply(tf.expand_dims(weight, axis=-1), kernel)

    neighboor_coords = tf.math.mod(neighboor_coords, nc)

    # Adding batch dimension to the neighboor coordinates
    batch_idx = tf.cast(tf.range(0, batch_size), dtype=tf.float32)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx,
                [1] + list(neighboor_coords.get_shape()[1:-1]) + [1])
    neighboor_coords = tf.concat([b, neighboor_coords], axis=-1)

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)

    update = tf.scatter_nd(tf.reshape(neighboor_coords, (-1, 8, 4)),
                           tf.reshape(kernel, (-1, 8)),
                           [batch_size, nx, ny, nz])
    mesh = mesh + update
    return mesh


def cic_paint_2d(mesh, part, weight=None, mask=None, name="CiCPaint2D"):
  """
  Paints particules on a 2D mesh.

  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc)
    Input 2D mesh tensor

  part: tensor (batch_size, npart, 2)
    List of 2D particle coordinates, assumed to be in mesh units.

  weight: tensor (batch_size, npart)
    List of weights  for each particle

  mask: tensor (batch_size, npart)
    Binary mask of whether particles should be included in the paint or not.
  """
  with tf.name_scope(name):
    mesh = tf.convert_to_tensor(mesh, name="mesh")
    part = tf.convert_to_tensor(part, name="part")
    if weight is not None:
      weight = tf.convert_to_tensor(weight, name="weight")
    if mask is not None:
      mask = tf.convert_to_tensor(mask, name="mask")

    shape = tf.shape(mesh)
    batch_size, nx, ny = shape[0], shape[1], shape[2]
    nc = ny

    # Flatten part if it's not already done
    if len(part.shape) > 2:
      part = tf.reshape(part, (batch_size, -1, 2))

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(
        tf.constant([[[0, 0], [1., 0], [0., 1], [1., 1]]]), 0)

    neighboor_coords = floor + connection
    kernel = 1. - tf.abs(part - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1]

    if weight is not None:
      kernel = tf.multiply(tf.expand_dims(weight, axis=-1), kernel)
    kernel = tf.reshape(kernel, (-1, 4))

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)
    neighboor_coords = tf.math.mod(neighboor_coords, nc)

    # Adding batch dimension to the neighboor coordinates
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx,
                [1] + list(neighboor_coords.get_shape()[1:-1]) + [1])
    neighboor_coords = tf.concat([b, neighboor_coords], axis=-1)
    neighboor_coords = tf.reshape(neighboor_coords, (-1, 4, 3))

    # Only keep the particles we want to keep if mask is supplied
    if mask is not None:
      mask = tf.where(tf.reshape(mask, [-1]))
      neighboor_coords = tf.gather(neighboor_coords, tf.reshape(mask, [-1]))
      kernel = tf.gather(kernel, tf.reshape(mask, [-1]))

    update = tf.scatter_nd(neighboor_coords, kernel, [batch_size, nx, ny])
    mesh = mesh + update
    return mesh


def cic_readout(mesh, part, name="CiCReadout"):
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
  with tf.name_scope("CiCReadout"):
    mesh = tf.convert_to_tensor(mesh, name="mesh")
    part = tf.convert_to_tensor(part, name="part")

    shape = tf.shape(mesh)
    batch_size, nx, ny, nz = shape[0], shape[1], shape[2], shape[3]
    nc = [nx, ny, nz]

    # Flatten part if it's not already done
    if len(part.shape) > 3:
      part = tf.reshape(part, (batch_size, -1, 3))

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(
        tf.constant([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                      [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]]), 0)

    neighboor_coords = tf.add(floor, connection)
    kernel = 1. - tf.abs(part - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords = tf.math.mod(neighboor_coords, nc)

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)
    meshvals = tf.gather_nd(mesh, neighboor_coords, batch_dims=1)
    weightedvals = tf.multiply(meshvals, kernel)
    value = tf.reduce_sum(weightedvals, axis=-1)
    return value


def r2c3d(rfield, norm=None, dtype=tf.complex64, name="R2C3D"):
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
  with tf.name_scope(name):
    rfield = tf.convert_to_tensor(rfield, name="mesh")
    if norm is None:
      norm = tf.cast(tf.reduce_prod(rfield.get_shape()[1:]), dtype)
    else:
      norm = tf.cast(norm, dtype)
    cfield = tf.multiply(tf.signal.fft3d(tf.cast(rfield, dtype)),
                         1 / norm,
                         name=name)
    return cfield


def c2r3d(cfield, norm=None, dtype=tf.float32, name="C2R3D"):
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
  with tf.name_scope(name):
    cfield = tf.convert_to_tensor(cfield, name="mesh")
    if norm is None:
      norm = tf.cast(tf.reduce_prod(cfield.get_shape()[1:]), dtype)
    else:
      norm = tf.cast(norm, dtype)
    rfield = tf.multiply(tf.cast(tf.signal.ifft3d(cfield), dtype),
                         norm,
                         name=name)
    return rfield


def r2c2d(rfield, norm=None, dtype=tf.complex64, name="R2C2D"):
  """
  Converts a real field to its complex Fourier Transform

  Parameters:
  -----------
  rfield: tensor (batch_size, nc, nc)
    Input 2D real field

  norm: float
    Normalization factor

  dtype: tf.dtype
    Type of output tensor

  Return:
  -------
  cfield: tensor (batch_size, nc, nc)
    Complex field
  """
  with tf.name_scope(name):
    rfield = tf.convert_to_tensor(rfield, name="mesh")
    if norm is None:
      norm = tf.cast(tf.reduce_prod(rfield.get_shape()[1:]), dtype)
    else:
      norm = tf.cast(norm, dtype)
    cfield = tf.multiply(tf.signal.fft2d(tf.cast(rfield, dtype)),
                         1 / norm,
                         name=name)
    return cfield


def c2r2d(cfield, norm=None, dtype=tf.float32, name="C2R2D"):
  """
  Converts a complex Fourier domain field to a real field

  Parameters:
  -----------
  cfield: tensor (batch_size, nc, nc)
    Complex 2D real field

  norm: float
    Normalization factor

  dtype: tf.dtype
    Type of output tensor

  Return:
  -------
  rfield: tensor (batch_size, nc, nc)
    Real valued field
  """
  with tf.name_scope(name):
    cfield = tf.convert_to_tensor(cfield, name="mesh")
    if norm is None:
      norm = tf.cast(tf.reduce_prod(cfield.get_shape()[1:]), dtype)
    else:
      norm = tf.cast(norm, dtype)
    rfield = tf.multiply(tf.cast(tf.signal.ifft2d(cfield), dtype),
                         norm,
                         name=name)
    return rfield


def white_noise(nc,
                batch_size=1,
                seed=None,
                type='complex',
                name="WhiteNoise"):
  """
  Samples a 3D cube of white noise of desired size
  """
  with tf.name_scope(name):
    # Transform nc to a list of necessary
    if isinstance(nc, int):
      nc = [nc, nc, nc]

    white = tf.random.normal(shape=[batch_size] + nc,
                             mean=0.,
                             stddev=(nc[0] * nc[1] * nc[2])**0.5,
                             seed=seed)
    if type == 'real':
      return white
    elif type == 'complex':
      whitec = r2c3d(white, norm=nc[0] * nc[1] * nc[2])
      return whitec
