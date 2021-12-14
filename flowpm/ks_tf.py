import numpy as np
import tensorflow as tf


def ks93_tf(g1, g2):
    """Direct inversion of weak-lensing shear to convergence.
    This function is an implementation of the Kaiser & Squires (1993) mass
    mapping algorithm. Due to the mass sheet degeneracy, the convergence is
    recovered only up to an overall additive constant. It is chosen here to
    produce output maps of mean zero. The inversion is performed in Fourier
    space for speed.
    Parameters
    ----------
    g1, g2 :  2-D Tensor 
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
    """
    # Check consistency of input maps
    assert g1.shape == g2.shape

    # Compute Fourier space grids
    (nx, ny) = g1.shape
    k1, k2 = tf.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    g1hat = tf.signal.fft2d(tf.cast(g1, dtype=tf.complex64))
    g2hat = tf.signal.fft2d(tf.cast(g2, dtype=tf.complex64))

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    mask = np.zeros(k2.shape)
    mask[0, 0] = 1
    k2 = k2 + tf.convert_to_tensor(mask)
    p1 = tf.cast(p1, dtype=tf.complex64)
    p2 = tf.cast(p2, dtype=tf.complex64)
    k2 = tf.cast(k2, dtype=tf.complex64)
    kEhat = (p1 * g1hat + p2 * g2hat) / k2
    kBhat = -(p2 * g1hat - p1 * g2hat) / k2

    # Transform back to pixel space
    kE = tf.math.real(tf.signal.ifft2d(kEhat))
    kB = tf.math.real(tf.signal.ifft2d(kBhat))

    return kE, kB

def ks93inv_tf(kE, kB):
    """Direct inversion of weak-lensing convergence to shear.
    This function provides the inverse of the Kaiser & Squires (1993) mass
    mapping algorithm, namely the shear is recovered from input E-mode and
    B-mode convergence maps.
    Parameters
    ----------
    kE, kB : 2-D Tensor 
        2D input arrays corresponding to the E-mode and B-mode (i.e., real and
        imaginary) components of convergence.
    Returns
    -------
    g1, g2 : tuple of numpy arrays
        Maps of the two components of shear.
    Raises
    ------
    AssertionError
        For input arrays of different sizes.
    See Also
    --------
    ks93
        For the forward operation (shear to convergence).
    """
    # Check consistency of input maps
    assert kE.shape == kB.shape

    # Compute Fourier space grids
    (nx, ny) = kE.shape
    k1, k2 = tf.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    # Compute Fourier transforms of kE and kB
    kEhat = tf.signal.fft2d(tf.cast(kE, dtype=tf.complex64))
    kBhat = tf.signal.fft2d(tf.cast(kB, dtype=tf.complex64))

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    mask = np.zeros(k2.shape)
    mask[0, 0] = 1
    k2 = k2 + tf.convert_to_tensor(mask)
    p1 = tf.cast(p1, dtype=tf.complex64)
    p2 = tf.cast(p2, dtype=tf.complex64)
    k2 = tf.cast(k2, dtype=tf.complex64)
    g1hat = (p1 * kEhat - p2 * kBhat) / k2
    g2hat = (p2 * kEhat + p1 * kBhat) / k2

    # Transform back to pixel space
    g1 =tf.math.real(tf.signal.ifft2d(g1hat))
    g2 =tf.math.real(tf.signal.ifft2d(g2hat))

    return g1, g2
