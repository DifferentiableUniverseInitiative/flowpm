"""Required Additional Mesh TensorFlow ops"""
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
