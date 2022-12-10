import tensorflow as tf
import flowpm.scipy.integrate as integrate
import flowpm.scipy.interpolate as interpolate
import flowpm
from astropy.io import fits
from flowpm.NLA_IA import k_IA
import astropy.units as u
import numpy as np


def LSST_Y1_tomog(cosmology,
                  lensplanes,
                  box_size,
                  z_source,
                  z,
                  nz,
                  field_npix,
                  field_size,
                  nbin,
                  batch_size=1,
                  use_A_ia=False,
                  Aia=None):
  """This function takes as input a list of lensplanes and redshift distribution and returns a stacked convergence maps for each tomographic bin) 
    
    Parameters:
    -----------
    
    cosmology: `Cosmology`,
        cosmology object.
    
    lensplanes: list of tuples (r, a, density_plane),
        lens planes to use
    
    boxsize: float 
        Transverse comoving size of the simulation volume [Mpc/h]

    z_source: array_like or tf.TensorArray
       Redshift of the source plane
       
    z: array_like or tf.TensorArray
       Redshift-coordinates where the n(z) is evaluated
    
    nz: array_like or tf.TensorArray of shape ([nbin, z])
        User-defined n(z) distribution.
    
    field_npix: Int
        Resolution of the final interpolated plane
    
    nbin: float.
        Number of photometric bins to use.
        
    batch_size: int
        Size of batches
        
    use_A_ia: Boolean
        If true, the frunction will return the stack convergence map for the IA signal,
        if false, the stack convergence map for the lensing signal
    
    Aia: Float or None (default)
        Amplitude parameter AI, describes the  strength of the tidal coupling. 
    
    Returns
    -------
    tom_kappa: tf.TensorArray [nbins,field_npix,field_npix]
    Stacked convergence maps for each tomographic bin
    
    Note:
    -------
    Details of the redshift distribution used can be found in this paper: https://arxiv.org/pdf/2111.04917.pdf 
    """

  xgrid, ygrid = np.meshgrid(
      np.linspace(0, field_size, field_npix,
                  endpoint=False),  # range of X coordinates
      np.linspace(0, field_size, field_npix,
                  endpoint=False))  # range of Y coordinates
  coords = np.stack([xgrid, ygrid], axis=0) * u.deg
  c = coords.reshape([2, -1]).T.to(u.rad)
  if use_A_ia is not False:
    sum_kappa = []
    for j in range(len(z_source)):
      im_IA = flowpm.raytracing.interpolation(
          lensplanes[j][-1],
          dx=box_size / 2048,
          r_center=lensplanes[j][0],
          field_npix=field_npix,
          coords=c)
      k_ia = k_IA(cosmology, lensplanes[j][1], im_IA, Aia)
      sum_kappa.append(k_ia[0])
    tom_kappa = [
        integrate.trapz(
            tf.reshape(
                interpolate.interp_tf(z_source, z,
                                      tf.cast(nz[i], dtype=tf.float32)),
                [-1, 1, 1]) * sum_kappa, z_source) for i in range(nbin)
    ]
  else:
    m = flowpm.raytracing.convergenceBorn(
        cosmology,
        lensplanes,
        dx=box_size / 2048,
        dz=box_size,
        coords=c,
        z_source=z_source,
        field_npix=field_npix)
    tom_kappa = [
        integrate.trapz(
            tf.reshape(interpolate.interp_tf(z_source, z, nz[i]), [-1, 1, 1]) *
            m, z_source) for i in range(nbin)
    ]
  return tom_kappa


def systematic_shift(z, bias):
  """Implements a systematic shift in a redshift distribution
    
    Parameters:
    -----------
    
    z: array_like or tf.TensorArray
       Photometric redshift array
        
             
    bias: float value
        Nuisance parameters defining the uncertainty of the redshift distributions
    
    """
  z = tf.convert_to_tensor(z, dtype=tf.float32)
  return (tf.clip_by_value(z - bias, 0, 50))
