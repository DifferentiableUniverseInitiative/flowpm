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
