"""Required Additional Mesh TensorFlow ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

def halo_reduce(x, blocks_dim, block_size_dim, halo_size, wrap=True):
  """Reduce each block with the margins of adjacent blocks.

  Get left and right blocks_dim and sum overlap along block_size_dim.
  Only supports halo size smaller than block_size/2

  Args:
    x: a Tensor.
    blocks_dim: a Dimension in x.shape
    block_size_dim: a Dimension in x.shape
    halo_size: an integer
    wrap: a boolean

  Returns:
    a Tensor with the same shape as x, other than in block_size_dim, whose
    size is increased by 2*halo_size.
  """
  if halo_size == 0:
    return x
  block_size = block_size_dim.size
  assert halo_size <= block_size//2

  left_margin = mtf.slice(x, 0, 2*halo_size, block_size_dim.name)
  right_margin = mtf.slice(x, block_size_dim.size - 2*halo_size, 2*halo_size, block_size_dim.name)
  center =  mtf.slice(x, 2*halo_size, block_size_dim.size - 4*halo_size, block_size_dim.name)

  # Perform halo exchange sum margins
  left =  mtf.shift(right_margin, 1, blocks_dim, wrap) + left_margin
  right = mtf.shift(left_margin, -1, blocks_dim, wrap) + right_margin

  # Recompose block
  left = mtf.pad(left, [0, block_size_dim.size- 2*halo_size], block_size_dim.name)
  right = mtf.pad(right, [block_size_dim.size- 2*halo_size, 0], block_size_dim.name)
  center = mtf.pad(center, [2*halo_size, 2*halo_size], block_size_dim.name)
  x = left + center + right
  return x

def fft3d(x):
  """
  Computes a 3D FFT along the 3 inner most dimensions

  This requires the last dimension to not be splitted
  """
  x = mtf.cast(x, tf.complex64)
  outer_dims =  x.shape[:-3]
  original_shape = x.shape
  # Loop over the number of dimensions
  for d in range(3):
    x_dim, y_dim, z_dim = x.shape[-3:]
    x = mtf.slicewise(tf.signal.fft, [x],
                      output_dtype=tf.complex64,
                      splittable_dims=x.shape[:-1])
    if d < 2:
      x = mtf.transpose(x, new_shape=outer_dims+[y_dim, z_dim, x_dim])
      x = mtf.reshape(x, new_shape=outer_dims+[y_dim, x_dim, z_dim])

  x = mtf.transpose(x, new_shape=outer_dims+[y_dim, z_dim, x_dim])
  x = mtf.reshape(x, new_shape=original_shape)
  return x

def ifft3d(x):
  """
  Computes an inverse 3D FFT along the 3 inner-most dimensions of x

  This requires the last dimension to not be splitted
  """
  x = mtf.cast(x, tf.complex64)
  outer_dims =  x.shape[:-3]
  original_shape = x.shape
  # Loop over the number of dimensions
  for d in range(3):
    x_dim, y_dim, z_dim = x.shape[-3:]
    x = mtf.slicewise(tf.signal.ifft, [x],
                      output_dtype=tf.complex64,
                      splittable_dims=x.shape[:-1])
    if d < 2:
      x = mtf.transpose(x, new_shape=outer_dims+[y_dim, z_dim, x_dim])
      x = mtf.reshape(x, new_shape=outer_dims+[y_dim, x_dim, z_dim])

  x = mtf.transpose(x, new_shape=outer_dims+[y_dim, z_dim, x_dim])
  x = mtf.reshape(x, new_shape=original_shape)
  return x

def random_normal(mesh, shape, **kwargs):
  """Random normal.

  Args:
    mesh: a Mesh
    shape: a Shape
    **kwargs: keyword args for tf.random.normal, except seed

  Returns:
    a Tensor
  """
  shape = mtf.convert_to_shape(shape)
  return mtf.RandomOperation(mesh, shape, tf.random.normal, **kwargs).outputs[0]
