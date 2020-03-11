""" Composite FastPM elements, implemented in Mesh Tensorflow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

import mesh_tensorflow as mtf

from . import mesh_ops
from . import mesh_utils
from . import mesh_kernels



def fr_to_hr(field, hr_shape, halo_size, splittables, mesh):
    # Reshaping array into high resolution mesh
    field = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),axis=1),
                          [field],
                          output_dtype=field.dtype,
                          output_shape=hr_shape,
                          name='my_reshape',
                          splittable_dims=splittables)

    for block_size_dim in hr_shape[-3:]:
        field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)

    for blocks_dim, block_size_dim in zip(hr_shape[1:4], field.shape[-3:]):
        field = mesh_ops.halo_reduce(field, blocks_dim, block_size_dim, halo_size)

    return field


def downsample_hr_to_lr(field, lr_shape, hr_shape, downsampling_factor, halo_size, splittables, mesh):
    # Reshaping array into high resolution mesh
    field = mtf.reshape(field, field.shape+[mtf.Dimension('h_dim', 1)])
    low = mesh_utils.downsample(field, downsampling_factor, antialias=True)
    low = mtf.reshape(low, low.shape[:-1])

    for block_size_dim in hr_shape[-3:]:
        low = mtf.slice(low, halo_size//2**downsampling_factor, block_size_dim.size//2**downsampling_factor, block_size_dim.name)
    # Hack usisng  custom reshape because mesh is pretty dumb
    low = mtf.slicewise(lambda x: x[:,0,0,0],
                        [low],
                        output_dtype=field.dtype,
                        output_shape=lr_shape,
                        name='my_dumb_reshape',
                        splittable_dims=splittables)

    return low


def cic_paint_fr(field, state, output_shape, hr_shape, halo_size, splittables, mesh):
    '''paint the position from state to a field of batch+3D tensor
    Ops performed :
    - reshape to hr_shape
    - pad
    - paint
    - halo_reduce
    - slice to remove pad
    - reshape to output_shape
    '''
    lnc = field.shape[-1].size
    field = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),axis=1),
                          [field],
                          output_dtype=tf.float32,
                          output_shape=mtf.Shape(hr_shape[0:4]+[
                              mtf.Dimension('sx_block', lnc//hr_shape[1].size),
                              mtf.Dimension('sy_block', lnc//hr_shape[2].size),
                              mtf.Dimension('sz_block', lnc//hr_shape[3].size)]),
                          name='my_reshape',
                          splittable_dims=splittables)

    for block_size_dim in hr_shape[-3:]:
        field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)

    field = mesh_utils.cic_paint(field, state[0], halo_size)
    # Halo exchange
    for blocks_dim, block_size_dim in zip(hr_shape[1:4], field.shape[-3:]):
        field = mesh_ops.halo_reduce(field, blocks_dim, block_size_dim, halo_size)
    # Remove borders
    for block_size_dim in hr_shape[-3:]:
        field = mtf.slice(field, halo_size, block_size_dim.size, block_size_dim.name)

    field = mtf.slicewise(lambda x: x[:,0,0,0],
                        [field],
                          output_dtype=field.dtype,
                          output_shape=output_shape,
                          name='my_dumb_reshape',
                          splittable_dims=splittables)
    return field





def cic_readout_fr(field, state, output_shape, hr_shape, halo_size, splittables, mesh):
    '''readout from at the position from state on a field of batch+3D tensor'''
    lnc = field.shape[-1].size
    field = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),axis=1),
                          [field],
                          output_dtype=tf.float32,
                          output_shape=mtf.Shape(hr_shape[0:4]+[
                              mtf.Dimension('sx_block', lnc//hr_shape[1].size),
                              mtf.Dimension('sy_block', lnc//hr_shape[2].size),
                              mtf.Dimension('sz_block', lnc//hr_shape[3].size)]),
                          name='my_reshape',
                          splittable_dims=splittables)

    for block_size_dim in hr_shape[-3:]:
        field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)
    #Halo exchange
    for blocks_dim, block_size_dim in zip(hr_shape[1:4], field.shape[-3:]):
        field = mesh_ops.halo_reduce(field, blocks_dim, block_size_dim, halo_size)

    read = mesh_utils.cic_readout(field, state[0], halo_size)
    return read
