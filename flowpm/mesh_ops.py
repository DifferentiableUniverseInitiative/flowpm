"""FastPM elements implemented with mesh tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow as tf

class IndicesOperation(mtf.Operation):
  """Distributed equivalent of np.indices"""

  def __init__(self, mesh, shape, dtype, name=None):
    super(IndicesOperation, self).__init__([], mesh, name=name or "indices")
    self._mesh = mesh
    self._shape = [mtf.convert_to_dimension(dim) for dim in shape]
    self._dtype = dtype
    self._outputs = [mtf.Tensor(self, mtf.Shape(self._shape + [mtf.Dimension("ndim", len(self._shape))]), dtype)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    sshape = mesh_impl.slice_shape(self.outputs[0].shape[:-1])
    def _tf_fn():
      tf_indices = [tf.range(dim) for dim in sshape]
      return tf.cast(tf.stack(tf.meshgrid(*tf_indices, indexing='ij'), axis=-1), dtype=self._dtype)
    value = mesh_impl.slicewise(_tf_fn)
    lowering.set_tensor_lowering(self.outputs[0], value)

def mtf_indices(mesh, shape, dtype, name=None):
  return IndicesOperation(mesh, shape, dtype, name).outputs[0]

def fft3d(x):
  """
  Computes a 3D FFT along the 3 inner most dimensions

  This requires the last dimension to not be splitted
  """
  x = mtf.cast(x, tf.complex64)
  outer_dims =  x.shape[:-3]
  x_dim, y_dim, z_dim = x.shape[-3:]
  # Loop over the number of dimensions
  for d in range(3):
    x = mtf.slicewise(tf.signal.fft, [x],
                      output_dtype=tf.complex64,
                      splittable_dims=x.shape[:-1])
    x = mtf.transpose(x, new_shape=outer_dims+[y_dim, z_dim, x_dim])
    x = mtf.replace_dimensions(x, [y_dim, z_dim, x_dim], [x_dim, y_dim, z_dim])
  return x

def ifft3d(x):
  """
  Computes an inverse 3D FFT along the 3 inner-most dimensions of x

  This requires the last dimension to not be splitted
  """
  x = mtf.cast(x, tf.complex64)
  outer_dims =  x.shape[:-3]
  x_dim, y_dim, z_dim = x.shape[-3:]
  # Loop over the number of dimensions
  for d in range(3):
    x = mtf.slicewise(tf.signal.ifft, [x],
                      output_dtype=tf.complex64,
                      splittable_dims=x.shape[:-1])
    x = mtf.transpose(x, new_shape=outer_dims+[y_dim, z_dim, x_dim])
    x = mtf.replace_dimensions(x, [y_dim, z_dim, x_dim], [x_dim, y_dim, z_dim])
  return x

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
    batch_size, nx, ny, nz = shape[0], shape[1], shape[2], shape[3]
    nc = nz

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
    # Replacing the reduce_prod op by manual multiplication
    # TODO: figure out why reduce_prod was crashing the Hessian computation
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    if weight is not None: kernel = tf.multiply(tf.expand_dims(weight, axis=-1) , kernel)

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)

    # Adding batch dimension to the neighboor coordinates
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, [1] + list(neighboor_coords.get_shape()[1:-1]) + [1])
    neighboor_coords = tf.concat([b, neighboor_coords], axis=-1)
    return tf.reshape(neighboor_coords, (batch_size, nx, ny, nz, 8,4)), tf.reshape(kernel, (batch_size, nx, ny, nz, 8))

def _cic_update(mesh, neighboor_coords, kernel, shift, name=None):
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
    batch_size, nx, ny, nz = shape[0], shape[1], shape[2], shape[3]
    nc = nz

    #TODO: Assert shift shape
    neighboor_coords = tf.reshape(neighboor_coords, (-1, 8,4))
    neighboor_coords = neighboor_coords + tf.reshape(tf.constant(shift), [1,1,4])
    neighboor_coords = tf.math.mod(neighboor_coords , nc)

    update = tf.scatter_nd(neighboor_coords,
                           tf.reshape(kernel, (-1, 8)),
                           [batch_size, nx, ny, nz])
    mesh = mesh + update
    return mesh

def cic_paint(mesh, part, splitted_dims, nsplits, weight=None, name=None):
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
                         splittable_dims=mesh.shape[:-1])

  dim_size = splitted_dims[0].size
  slice_size = dim_size // nsplits[0]

  # Implement simple split along one axis
  if len(splitted_dims) == 1:
    mesh = mtf.slicewise(lambda x,y,z: _cic_update(x, y, z, shift=[0, slice_size//2, 0, 0]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=mesh.shape,
                         splittable_dims=mesh.shape[:-1])
    mesh = mtf.shift(mesh, -slice_size, mesh.shape[-3], wrap=True)
    mesh = mtf.slicewise(lambda x,y,z: _cic_update(x, y, z, shift=[0, -slice_size//2, 0, 0]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=mesh.shape,
                         splittable_dims=mesh.shape[:-1])
    mesh = mtf.shift(mesh, slice_size//2, mesh.shape[-3], wrap=True)

  elif len(splitted_dims) == 2:
    mesh = mtf.slicewise(lambda x,y,z: _cic_update(x, y, z, shift=[0, slice_size//2, slice_size//2, 0]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=mesh.shape,
                         splittable_dims=mesh.shape[:-1])
    mesh = mtf.shift(mesh, -slice_size, mesh.shape[-2], wrap=True)
    mesh = mtf.slicewise(lambda x,y,z: _cic_update(x, y, z, shift=[0, slice_size//2, -slice_size//2, 0]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=mesh.shape,
                         splittable_dims=mesh.shape[:-1])
    mesh = mtf.shift(mesh, slice_size//2, mesh.shape[-2], wrap=True)
    mesh = mtf.shift(mesh, -slice_size, mesh.shape[-3], wrap=True)
    mesh = mtf.slicewise(lambda x,y,z: _cic_update(x, y, z, shift=[0, -slice_size//2, slice_size//2, 0]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=mesh.shape,
                         splittable_dims=mesh.shape[:-1])
    mesh = mtf.shift(mesh, -slice_size, mesh.shape[-2], wrap=True)
    mesh = mtf.slicewise(lambda x,y,z: _cic_update(x, y, z, shift=[0, -slice_size//2, -slice_size//2, 0]),
                         [mesh, indices, values],
                         output_dtype=tf.float32,
                         output_shape=mesh.shape,
                         splittable_dims=mesh.shape[:-1])
    mesh = mtf.shift(mesh, slice_size//2, mesh.shape[-2], wrap=True)
    mesh = mtf.shift(mesh, slice_size//2, mesh.shape[-3], wrap=True)
  return mesh
