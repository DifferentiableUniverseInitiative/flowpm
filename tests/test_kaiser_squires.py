import tensorflow as tf
import numpy as np
import jax
import DifferentiableHOS
from numpy.testing import assert_allclose
from flowpm.ks_tf import ks93_tf, ks93inv_tf


def ks93(g1, g2):
  """Direct inversion of weak-lensing shear to convergence.
    This function is an implementation of the Kaiser & Squires (1993) mass
    mapping algorithm. Due to the mass sheet degeneracy, the convergence is
    recovered only up to an overall additive constant. It is chosen here to
    produce output maps of mean zero. The inversion is performed in Fourier
    space for speed.
    Parameters
    ----------
    g1, g2 : array_like
        2D input arrays corresponding to the first and second (i.e., real and
        imaginary) components of shear, binned spatially to a regular grid.
    Returns
    -------
    kE, kB : tuple of numpy arrays
        E-mode and B-mode maps of convergence.
    Raises
    ------
    AssertionError
        For input arrays of different sizes.
    See Also
    --------
    bin2d
        For binning a galaxy shear catalog.
    Examples
    --------
    >>> # (g1, g2) should in practice be measurements from a real galaxy survey
    >>> g1, g2 = 0.1 * np.random.randn(2, 32, 32) + 0.1 * np.ones((2, 32, 32))
    >>> kE, kB = ks93(g1, g2)
    >>> kE.shape
    (32, 32)
    >>> kE.mean()
    1.0842021724855044e-18
    """
  # Check consistency of input maps
  assert g1.shape == g2.shape

  # Compute Fourier space grids
  (nx, ny) = g1.shape
  k1, k2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

  # Compute Fourier transforms of g1 and g2
  g1hat = np.fft.fft2(g1)
  g2hat = np.fft.fft2(g2)

  # Apply Fourier space inversion operator
  p1 = k1 * k1 - k2 * k2
  p2 = 2 * k1 * k2
  k2 = k1 * k1 + k2 * k2
  #k2[0, 0] = 1  # avoid division by 0
  k2 = jax.ops.index_update(k2, jax.ops.index[0, 0], 1.)  # avoid division by 0
  kEhat = (p1 * g1hat + p2 * g2hat) / k2
  kBhat = -(p2 * g1hat - p1 * g2hat) / k2

  # Transform back to pixel space
  kE = np.fft.ifft2(kEhat).real
  kB = np.fft.ifft2(kBhat).real

  return kE, kB


def test_ks93():
  """ Testing tensorflow implementation of Kaiser & Squires (1993 vs. numpy implementation """
  e1, e2 = np.load("e.npy")
  ke, kb = ks93(e1, e2)
  ke_tf, kb_tf = ks93_tf(e1, e2)

  assert_allclose(ke_tf.numpy(), ke, atol=1e-5)
  assert_allclose(kb_tf.numpy(), kb, atol=1e-5)
  print("Kaiser & Squires test complete")
