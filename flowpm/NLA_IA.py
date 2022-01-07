from flowpm.tfbackground import D1
import flowpm.constants as constants
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import flowpm
from flowpm.fourier_smoothing import fourier_smoothing


def Epsilon(cosmology, a_source, tidal_field, Aia):
  r""" Compute the real and the imaginary component of the intrinsic ellipticities

  Parameters
  ----------
    cosmology: `Cosmology`,
        cosmology object.
        
    a_source : 1-D tf.TensorArray
        Source scale factor    
     
    tidal_field:  Tensor of shape ([3, tidal_field_npix, tidal_field_npix]),
        Interpolated projected tidal shear of the source plane 
        
    Aia: Float.
        Amplitude parameter AI, describes the  strength of the tidal coupling. 
    
    
  Returns
  -------
    e1: 2-D Tensor.
        Real component of the intrinsic ellipticities.
        
    e2: 2-D Tensor.
        Imaginary component of the intrinsic ellipticities.
  """
  Omega_m = cosmology.Omega_c + cosmology.Omega_b
  norm_factor = -Aia * constants.C_1 * Omega_m * constants.rhocrit / D1(
      cosmology, a_source)
  e1 = norm_factor * (tidal_field[0] - tidal_field[1])
  e2 = 2 * norm_factor * (tidal_field[2])
  return e1, e2


def k_IA(cosmology, a_source, projected_density_plane, Aia):
  r""" Compute the convergence map of the NLA intrinsic ellipticities

  Parameters
  ----------
    cosmology: `Cosmology`,
        cosmology object.
        
    a_source : 1-D tf.TensorArray
        Source scale factor    
     
    Projected_density_plane: Tensor of Shape([batch_size, plane_resolution, plane_resolution])
        Density plane projected on the light cone   
        
    Aia: Float.
        Amplitude parameter AI, describes the  strength of the tidal coupling. 
    
    
  Returns
  -------
    k: Tensor of Shape ([batch_size, resolution, resolution]).
       Compute the convergence map of the NLA intrinsic ellipticities.

  """

  norm_factor = -Aia * constants.C_1 * cosmology.Omega_m * constants.rhocrit / D1(
      cosmology, a_source)
  k = norm_factor * projected_density_plane
  return k


def tidal_field(density_plane, resolution, sigma):
  r""" Compute the projected tidal shear

  Parameters
  ----------
  density_plane: Tensor of Shape([batch_size, plane_resolution, plane_resolution])
    Projected density 

  resolution : Int
      Pixel resolution of the input density plane 
      
  sigma: Float
      Sigma of the two-dimensional smoothing kernel in units of pixels

  Returns
  -------
  tidal_planes : Tensor of shape [3, plane_resolution,plane_resolution] (sx, sy, sz)
      Projected tidal shear of the source plane

  Notes
  -----
  The parametrization :cite:`J. Harnois-De ́raps et al.` for the projected tidal shear
   is given by:
   
  .. math::

   \tilde{s}_{ij,2D}(\textbf{k}_{\bot})=
   2 \pi \left [ \frac{k_ik_j}{k^2}- \frac{1}{3} \right ]
   \tilde{\delta}_{2D}(\textbf{k}_{\bot})
   \mathcal{G}_{2D}(\sigma_g)

  """

  k = np.fft.fftfreq(resolution)
  kx, ky = np.meshgrid(k, k)
  k2 = kx**2 + ky**2
  k2[0, 0] = 1.
  sxx = (kx * kx / k2 - 1. / 3)
  sxx[0, 0] = 0.
  syy = (ky * ky / k2 - 1. / 3)
  syy[0, 0] = 0.
  sxy = (kx * ky / k2)
  sxy[0, 0] = 0.
  ss = tf.stack([sxx, syy, sxy], axis=0)
  ss = tf.cast(ss, dtype=tf.complex64)
  ftt_density_plane = flowpm.utils.r2c2d(density_plane)
  ss_fac = ss * ftt_density_plane
  ss_smooth = fourier_smoothing(ss_fac, sigma, resolution, complex_plane=True)
  tidal_planes = flowpm.utils.c2r2d((ss_smooth))
  return tidal_planes


def interpolation(density_plane, dx, r_source, field_npix, coords):
  r""" Compute the interpolation of the projected density plane on the light-cones
   Parameters
   ----------
    density_plane: Tensor of shape: [1, resolution, resolution] (or [3, resolution, resolution] (corrisponding to (sx, sy, sz)) if tidal field).
        Projected density plane
    
    dx: float 
        transverse pixel resolution of the density planes [Mpc/h]

    r_center: tf.Tensor 
        Center of the densityplane [Mpc/h]
    
    field_npix: Int
        Resolution of the final interpolated plane
    
    coords: 3-D array.
        Angular coordinates in radians of N points with shape [batch, N, 2].

    Returns
    -------
    im= Tensor of shape [3, field_npix,field_npix].
        Interpolated projected density plane on the light-cones
    """
  coords = tf.convert_to_tensor(coords, dtype=tf.float32)
  c = coords * r_source / dx

  # Applying periodic conditions on sourceplane
  shape = tf.shape(density_plane)
  c = tf.math.mod(c, tf.cast(shape[1], tf.float32))

  # Shifting pixel center convention
  c = tf.expand_dims(c, axis=0) - 0.5

  im = tfa.image.interpolate_bilinear(
      tf.expand_dims(density_plane, -1), c, indexing='xy')
  im = tf.reshape(im, [shape[0], field_npix, field_npix])
  return im
