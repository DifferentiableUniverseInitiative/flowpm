from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from . import mesh_ops
from . import mesh_kernels

def _cic_indexing(mesh, part, weight=None, name=None):
  """
  Computes the indices and weights to use for painting the local particles on
  mesh, stops short of actually applying the painting.

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
  with tf.name_scope(name, "cic_indexing", [mesh, part, weight]):
    shape = tf.shape(mesh)
    batch_size =shape[0]
    nx, ny, nz = shape[-3], shape[-2], shape[-1]
    part_shape = part.shape

    # Flatten part if it's not already done
    if len(part.shape) > 3:
      part = tf.reshape(part, (batch_size, -1, 3))

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(tf.constant([[[0, 0, 0], [1., 0, 0],[0., 1, 0],
                                              [0., 0, 1],[1., 1, 0],[1., 0, 1],
                                              [0., 1, 1],[1., 1, 1]]]), 0)
    neighboor_coords = floor + connection
    kernel = 1. - tf.abs(part - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    if weight is not None: kernel = tf.multiply(tf.expand_dims(weight, axis=-1) , kernel)

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)

    # Adding batch dimension to the neighboor coordinates
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, [1] + list(neighboor_coords.get_shape()[1:-1]) + [1])
    neighboor_coords = tf.concat([b, neighboor_coords], axis=-1)
    return tf.reshape(neighboor_coords, part_shape[:-1]+[8, 4]),  tf.reshape(kernel, part_shape[:-1]+[8])

def _cic_paint(mesh, neighboor_coords, kernel, shift, name=None):
  """
  Paints particules on a 3D mesh.

  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc, nc)
    Input 3D mesh tensor

  shift: [x,y,z] array of coordinate shifting
  """
  with tf.name_scope(name, "cic_update", [mesh, neighboor_coords, kernel]):
    shape = tf.shape(mesh)
    batch_size = shape[0]
    nx, ny, nz = shape[-3], shape[-2], shape[-1]

    #TODO: Assert shift shape
    neighboor_coords = tf.reshape(neighboor_coords, (-1, 8,4))
    neighboor_coords = neighboor_coords + tf.reshape(tf.constant(shift), [1,1,4])

    update = tf.scatter_nd(neighboor_coords,
                           tf.reshape(kernel, (-1, 8)),
                           [batch_size, nx, ny, nz])

    mesh = mesh + tf.reshape(update, mesh.shape)
    return mesh

def _cic_readout(mesh, neighboor_coords, kernel, shift, name=None):
  """
  Paints particules on a 3D mesh.

  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc, nc)
    Input 3D mesh tensor

  shift: [x,y,z] array of coordinate shifting
  """
  with tf.name_scope(name, "cic_readout", [mesh, neighboor_coords, kernel]):
    shape = tf.shape(mesh)
    batch_size = shape[0]
    nx, ny, nz = shape[-3], shape[-2], shape[-1]
    mesh = mesh[:,0,0,0]
    shape_part = tf.shape(neighboor_coords)

    #TODO: Assert shift shape
    neighboor_coords = tf.reshape(neighboor_coords, (-1, 8,4))
    neighboor_coords = neighboor_coords + tf.reshape(tf.constant(shift), [1,1,4])

    meshvals = tf.gather_nd(mesh, neighboor_coords)

    weightedvals = tf.multiply(meshvals, tf.reshape(kernel, (-1, 8)))

    value = tf.reduce_sum(weightedvals, axis=-1)

    value = tf.reshape(value, shape_part[:-2])
    return value

def cic_paint(mesh, part, halo_size, weight=None, name=None):
  """
  Distributed Cloud In Cell implementation.

  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc, nc)
    Input 3D mesh tensor

  part: tensor (batch_size, npart, 3)
    List of 3D particle coordinates, assumed to be in mesh units if
    boxsize is None

  splitted_dims: list of Dimensions in mesh
    List of dimensions along which the particles are splitted

  weight: tensor (batch_size, npart)
    List of weights  for each particle

  ----
  The current implementation applies slicewise CiC paintings, and send/recv the
  part tensor with neighboring devices on the mesh to make sure particles
  displaced outside of their initial boundaries get a chance to be painted.
  """
  nk = mtf.Dimension("nk", 8)
  nl = mtf.Dimension("nl", 4)
  indices, values = mtf.slicewise(_cic_indexing,
                         [mesh, part],
                         output_dtype=[tf.float32, tf.float32],
                         output_shape=[mtf.Shape(part.shape.dims[:-1]+[nk, nl]),
                                       mtf.Shape(part.shape.dims[:-1]+[nk])],
                         splittable_dims=mesh.shape[:-3]+part.shape[1:-1])

  mesh = mtf.slicewise(lambda x,y,z: _cic_paint(x, y, z, shift=[0, halo_size, halo_size, halo_size]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=mesh.shape,
                         splittable_dims=mesh.shape[:-3]+part.shape[1:-1])
  return mesh

def cic_readout(mesh, part, halo_size, name=None):
  nk = mtf.Dimension("nk", 8)
  nl = mtf.Dimension("nl", 4)

  indices, values = mtf.slicewise(_cic_indexing,
                         [mesh, part],
                         output_dtype=[tf.float32, tf.float32],
                         output_shape=[mtf.Shape(part.shape.dims[:-1]+[nk, nl]),
                                       mtf.Shape(part.shape.dims[:-1]+[nk])],
                         splittable_dims=mesh.shape[:-3]+part.shape[1:-1])

  value = mtf.slicewise(lambda x,y,z: _cic_readout(x, y, z, shift=[0, halo_size, halo_size, halo_size]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=part.shape[:-1],
                         splittable_dims=mesh.shape[:-3]+part.shape[1:-1])
  return value

def downsample(field, downsampling_factor=2, antialias=True):
  """
  Performs a multiresolution decomposition of the input field.

  The input field will be decomposed into a low resolution approximation,
  and a details component.
  """
  low = field
  for i in range(downsampling_factor):
    kernel = mesh_kernels.get_bspline_kernel(low, mtf.Dimension('down_%d'%i,low.shape[-1].size), order=6)
    low = mtf.Conv3dOperation(low, kernel, strides=(1,2,2,2,1), padding='SAME').outputs[0]
  if antialias:
    kernel = mesh_kernels.get_bspline_kernel(low, mtf.Dimension('dogwn_%d'%i,low.shape[-1].size), order=2)
    low = mtf.Conv3dOperation(low, kernel, strides=(1,1,1,1,1), padding='SAME').outputs[0]
  return low

def upsample(low, downsampling_factor=2):
  """
  Performs a multiresolution reconstruction of the input field.

  The input field will be decomposed into a low resolution approximation,
  and a details component.
  """
  for i in range(downsampling_factor):
    kernel = mesh_kernels.get_bspline_kernel(low, mtf.Dimension('out_%d'%i,low.shape[-1].size), transpose=True, order=6)
    low = mtf.Conv3dTransposeOperation(low, kernel * 2.0**3, strides=(1,2,2,2,1), padding='SAME').outputs[0]
  return low

def split_scales(field, downsampling_factor=2., antialias=True):
  """
  Performs a multiresolution decomposition of the input field.

  The input field will be decomposed into a low resolution approximation,
  and a details component.
  """
  low = downsample(field, downsampling_factor, antialias)
  high = upsample(low, downsampling_factor)
  high = field - mtf.reshape(high, field.shape)
  return low, high

def slicewise_r2c3d(rfield):
  cfield = mtf.slicewise(lambda x: tf.signal.fft3d(tf.cast(x, tf.complex64)), [rfield],
                         output_dtype=tf.complex64,
                         splittable_dims=rfield.shape[:-3])
  return cfield

def slicewise_c2r3d(cfield):
  rfield = mtf.slicewise(lambda x: tf.cast(tf.signal.ifft3d(x), tf.float32),
                         [cfield],
                         output_dtype=tf.float32,
                         splittable_dims=cfield.shape[:-3])
  return rfield

def r2c3d(rfield, k_dims, norm=None, dtype=tf.complex64):
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
  x_dim, y_dim, z_dim = rfield.shape[-3:]
  if norm is None:
    norm = mtf.constant(rfield.mesh, x_dim.size*y_dim.size*z_dim.size)
  cfield = mesh_ops.fft3d(mtf.cast(rfield / norm, dtype), k_dims)
  return cfield

def c2r3d(cfield, dims, norm=None, dtype=tf.float32, name=None):
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
  x_dim, y_dim, z_dim = cfield.shape[-3:]
  if norm is None:
    norm = mtf.constant(cfield.mesh, x_dim.size*y_dim.size*z_dim.size)
  rfield = mtf.cast(mesh_ops.ifft3d(cfield, dims), dtype) * norm
  return rfield
