import numpy as np
import numpy

import tensorflow as tf


def _initialize_pk(shape, boxsize, kmin, dk):
  """
       Helper function to initialize various (fixed) values for powerspectra... not differentiable!
    """
  I = np.eye(len(shape), dtype='int') * -2 + 1

  W = np.empty(shape, dtype='f4')
  W[...] = 2.0
  W[..., 0] = 1.0
  W[..., -1] = 1.0

  kmax = np.pi * np.min(shape.as_list()) / np.max(boxsize) + dk / 2
  kedges = np.arange(kmin, kmax, dk)

  k = [
      np.fft.fftfreq(N, 1. / (N * 2 * np.pi / L))[:pkshape].reshape(kshape)
      for N, L, kshape, pkshape in zip(shape, boxsize, I, shape)
  ]
  kmag = sum(ki**2 for ki in k)**0.5

  xsum = np.zeros(len(kedges) + 1)
  Nsum = np.zeros(len(kedges) + 1)

  dig = np.digitize(kmag.flat, kedges)

  xsum.flat += np.bincount(dig, weights=(W * kmag).flat, minlength=xsum.size)
  Nsum.flat += np.bincount(dig, weights=W.flat, minlength=xsum.size)
  dig = tf.convert_to_tensor(dig, dtype=tf.int32)
  Nsum = tf.convert_to_tensor(Nsum, dtype=tf.complex64)
  return dig, Nsum, xsum, W, k, kedges


def power_spectrum(field, kmin=5, dk=0.5, boxsize=False):
  """
    Calculate the powerspectra given real space field
    
    Args:
        
        field: real valued field 
        kmin: minimum k-value for binned powerspectra
        dk: differential in each kbin
        shape: shape of field to calculate field (can be strangely shaped?)
        boxsize: length of each boxlength (can be strangly shaped?)
    
    Returns:
        
        kbins: the central value of the bins for plotting
        power: real valued array of power in each bin
        
  """
  shape = field.get_shape()
  batch_size, nx, ny, nz = shape

  #initialze values related to powerspectra (mode bins and weights)
  dig, Nsum, xsum, W, k, kedges = _initialize_pk(shape[1:], boxsize, kmin, dk)

  #convert field to complex for fft
  field_complex = tf.dtypes.cast(field, dtype=tf.complex64)

  #fast fourier transform
  fft_image = tf.signal.fft3d(field_complex)

  #absolute value of fast fourier transform
  pk = tf.math.real(fft_image * tf.math.conj(fft_image))

  def fn(x):
    #calculating powerspectra
    real = tf.reshape(tf.math.real(x), [-1])
    imag = tf.reshape(tf.math.imag(x), [-1])

    Psum = tf.dtypes.cast(
        tf.math.bincount(
            dig, weights=(W.flatten() * imag), minlength=xsum.size),
        dtype=tf.complex64) * 1j
    Psum += tf.dtypes.cast(
        tf.math.bincount(
            dig, weights=(W.flatten() * real), minlength=xsum.size),
        dtype=tf.complex64)
    return tf.dtypes.cast(
        (Psum / Nsum)[1:-1] * boxsize.prod(), dtype=tf.float32)

  power = tf.map_fn(
      fn,
      pk,
      dtype=None,
      parallel_iterations=None,
      back_prop=True,
      swap_memory=False,
      infer_shape=True,
      name=None)

  #normalization for powerspectra
  norm = tf.dtypes.cast(tf.reduce_prod(shape[1:]), dtype=tf.float32)**2

  #find central values of each bin
  kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2

  return kbins, power / norm
