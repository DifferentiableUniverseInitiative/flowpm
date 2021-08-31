"""Required Additional Mesh TensorFlow ops"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import numpy as np


class IndicesOperation(mtf.Operation):
  """Distributed equivalent of np.indices"""

  def __init__(self, mesh, shape, dtype, name=None):
    super(IndicesOperation, self).__init__([], mesh, name=name or "indices")
    self._mesh = mesh
    self._shape = [mtf.convert_to_dimension(dim) for dim in shape]
    self._dtype = dtype
    self._outputs = [
        mtf.Tensor(
            self,
            mtf.Shape(self._shape + [mtf.Dimension("ndim", len(self._shape))]),
            dtype)
    ]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    sshape = mesh_impl.slice_shape(self.outputs[0].shape[:-1])

    def _tf_fn():
      tf_indices = [tf.range(dim) for dim in sshape]
      return tf.cast(
          tf.stack(tf.meshgrid(*tf_indices, indexing='ij'), axis=-1),
          dtype=self._dtype)

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
  assert halo_size <= block_size // 2

  left_margin = mtf.slice(x, 0, 2 * halo_size, block_size_dim.name)
  right_margin = mtf.slice(x, block_size_dim.size - 2 * halo_size,
                           2 * halo_size, block_size_dim.name)
  center = mtf.slice(x, 2 * halo_size, block_size_dim.size - 4 * halo_size,
                     block_size_dim.name)

  # Perform halo exchange sum margins
  left = mtf.shift(right_margin, 1, blocks_dim, wrap) + left_margin
  right = mtf.shift(left_margin, -1, blocks_dim, wrap) + right_margin

  # Recompose block
  left = mtf.pad(left, [0, block_size_dim.size - 2 * halo_size],
                 block_size_dim.name)
  right = mtf.pad(right, [block_size_dim.size - 2 * halo_size, 0],
                  block_size_dim.name)
  center = mtf.pad(center, [2 * halo_size, 2 * halo_size], block_size_dim.name)
  x = left + center + right
  return x


class FFT3DOperation(mtf.Operation):
  """
  Performs 3D FFT along the trailing dimensions.

  This returns a transposed FFT however, to save a few all2all
  communications.
  """

  def __init__(self, tensor_in, k_dims, name=None):
    super(FFT3DOperation, self).__init__([tensor_in], name=name or "FFT3D")
    self._k_dims = k_dims
    self._output_shape = mtf.Shape(tensor_in.shape[:-3] +
                                   [k_dims[1], k_dims[2], k_dims[0]])
    self._outputs = [
        mtf.Tensor(self, mtf.Shape(self._output_shape), tensor_in.dtype)
    ]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    x = self.inputs[0]
    return [ifft3d(dy, x.shape[-3:])]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    x = self.inputs[0]
    naxes = len(x.shape)
    slices = lowering.tensors[self.inputs[0]]
    # Before performing any operations, we check the splitting
    split_axes = []
    for i in range(3):
      split_axes.append(
          mesh_impl.tensor_dimension_to_mesh_axis(x.shape.dims[-3:][i]))

    # Perform FFT followed by tranposes
    for i in range(2):
      # Apply FFT along last axis
      slices = mesh_impl.slicewise(tf.spectral.fft, slices)

      # Before transposing the array, making sure the new last dimension will
      # be contiguous
      if split_axes[-2] is not None:
        slices = mesh_impl.alltoall(slices, split_axes[-2], naxes - 1,
                                    naxes - 2)
        split_axes[-1] = split_axes[-2]
        split_axes[-2] = None
      perm = np.arange(len(x.shape))
      perm[-3:] = np.roll(perm[-3:], shift=1)
      slices = mesh_impl.slicewise(lambda x: tf.transpose(x, perm), slices)
      split_axes = [split_axes[2], split_axes[0], split_axes[1]]

    # Apply FFT along last axis
    slices = mesh_impl.slicewise(tf.spectral.fft, slices)
    lowering.set_tensor_lowering(self.outputs[0], slices)


def fft3d(x, k_dims):
  return FFT3DOperation(x, k_dims).outputs[0]


class iFFT3DOperation(mtf.Operation):
  """
  Performs inverse 3D FFT along the trailing dimensions.
  This assumes a transposed FFT as input however, to save a few all2all
  communications.
  """

  def __init__(self, tensor_in, dims, name=None):
    super(iFFT3DOperation, self).__init__([tensor_in], name=name or "iFFT3D")
    self._dims = dims
    self._output_shape = mtf.Shape(tensor_in.shape[:-3] + dims)
    self._outputs = [
        mtf.Tensor(self, mtf.Shape(self._output_shape), tensor_in.dtype)
    ]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    ky, kz, kx = self.inputs[0].shape[-3:]
    return [fft3d(dy, [kx, ky, kz])]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    x = self.inputs[0]
    naxes = len(x.shape)
    slices = lowering.tensors[self.inputs[0]]
    # Before performing any operations, we check the splitting
    split_axes = []
    for i in range(3):
      split_axes.append(
          mesh_impl.tensor_dimension_to_mesh_axis(x.shape.dims[-3:][i]))

    # Perform FFT followed by tranposes
    for i in range(2):
      # Apply FFT along last axis
      slices = mesh_impl.slicewise(tf.spectral.ifft, slices)

      # Before transposing the array, making sure the new last dimension will
      # be contiguous
      if split_axes[0] is not None:
        slices = mesh_impl.alltoall(slices, split_axes[0], naxes - 1, naxes - 3)
        split_axes[-1] = split_axes[0]
        split_axes[0] = None
      perm = np.arange(len(x.shape))
      perm[-3:] = np.roll(perm[-3:], shift=-1)
      slices = mesh_impl.slicewise(lambda x: tf.transpose(x, perm), slices)
      split_axes = [split_axes[1], split_axes[2], split_axes[0]]

    # Apply FFT along last axis
    slices = mesh_impl.slicewise(tf.spectral.ifft, slices)
    lowering.set_tensor_lowering(self.outputs[0], slices)


def ifft3d(x, k_dims):
  return iFFT3DOperation(x, k_dims).outputs[0]
