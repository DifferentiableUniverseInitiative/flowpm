from scipy.stats import norm
import numpy as np
import tensorflow as tf
import flowpm


def make_power_map(power_spectrum, size, kps=None):
  #Ok we need to make a map of the power spectrum in Fourier space
  k1 = np.fft.fftfreq(size)

  k2 = np.fft.fftfreq(size)

  kcoords = np.meshgrid(k1, k2)

  # Now we can compute the k vector
  k = np.sqrt(kcoords[0]**2 + kcoords[1]**2)

  if kps is None:
    kps = np.linspace(0, 0.5, len(power_spectrum))
  # And we can interpolate the PS at these positions

  ps_map = np.interp(k.flatten(), kps, power_spectrum).reshape([size, size])

  ps_map = tf.cast(ps_map, dtype=tf.float32)
  return ps_map


def fourier_smoothing(im, sigma, resolution, complex_plane=None):
  r""" Multidimensional Gaussian fourier filter.
    
    Parameters
    ----------
    
    im: tf.Tensor of real values if complex_plane=None, complex values if complex_plane=True
    The input array.

    sigma: float 
        The sigma of the Gaussian kernel. 

    resolution:Int
         Pixel resolution of the input image 
         
    complex_plane: boolean
    If false, the frunction will compute the Fourier transform of the input array
    before the convolution with the Gaussian filter,
    if true, the input array is already the Fourier transform of a real array.
         
    Returns
    -------
    im: tf.Tensor
     The filtered input.
    """
  if complex_plane is not None:
    im = im
  else:
    im = tf.signal.fft2d(tf.cast(im, tf.complex64))
  kps = np.linspace(0, 0.5, resolution)
  filter = norm(0, 1. / (2. * np.pi * sigma)).pdf(kps)
  m = make_power_map(filter, resolution, kps=kps)
  m /= m[0, 0]
  im = tf.cast(tf.reshape(m, [1, resolution, resolution]), tf.complex64) * im
  if complex_plane is not None:
    return im
  else:
    return tf.cast(tf.signal.ifft2d(im), tf.float32)
