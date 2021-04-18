# Testing raytracing functions against lenstools
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from scipy.ndimage import fourier_gaussian
import flowpm
import flowpm.constants as constants
import lenstools as lt
import bigfile

np.random.seed(0)

bs = 50
nc = 64
plane_resolution = 64

class FlowPMSnapshot(lt.simulations.nbody.NbodySnapshot):
  """
    A class that handles FlowPM simulation snapshots for lenstools
    """

  _header_keys = [
      'masses', 'num_particles_file', 'num_particles_total', 'box_size',
      'num_files', 'Om0', 'Ode0', 'h'
  ]

  ############################
  #Open the file with bigfile#
  ############################

  @classmethod
  def open(cls, filename, pool=None, header_kwargs=dict(), **kwargs):

    if bigfile is None:
      raise ImportError("bigfile must be installed!")

    fp = bigfile.BigFile(cls.buildFilename(filename, pool, **kwargs))
    return cls(fp, pool, header_kwargs=header_kwargs)

  ###################################################################################
  ######################Abstract method implementation###############################
  ###################################################################################

  @classmethod
  def buildFilename(cls, root, pool):
    return root

  @classmethod
  def int2root(cls, name, n):
    return name

  def getHeader(self):

    #Initialize header
    header = dict()
    bf_header = self.fp["Header"].attrs

    ###############################################
    #Translate FlowPM header into lenstools header#
    ###############################################

    #Number of particles/files
    header["num_particles_file"] = bf_header["NC"][0]**3
    header["num_particles_total"] = header["num_particles_file"]
    header["num_files"] = 1

    #Cosmology
    header["Om0"] = bf_header["OmegaCDM"][0] + bf_header["OmegaB"][0]
    header["Ode0"] = 1. - header["Om0"]
    header["w0"] = -1.
    header["wa"] = 0.
    header["h"] = bf_header["h"][0]
    header["redshift"] = 1/(bf_header["Time"][0]) - 1
    header["comoving_distance"] = bf_header["comoving_distance"][0] * 1.0e3
    header["scale_factor"] = bf_header["Time"][0]

    #Box size in kpc/h
    header["box_size"] = bf_header["BoxSize"][0] * 1.0e3
    header["box_size_mpch"] = bf_header["BoxSize"][0]
    #Plane Resolution
    header["nc"] = bf_header["NC"][0]
    #Masses
    header["masses"] = np.array(
        [0., bf_header["M0"][0] * header["h"], 0., 0., 0., 0.])
    #################

    return header

  def setLimits(self):

    if self.pool is None:
      self._first = None
      self._last = None
    else:

      #Divide equally between tasks
      Nt, Np = self.pool.size + 1, bigfile.BigData(self.fp).size
      part_per_task = Np // Nt
      self._first = part_per_task * self.pool.rank
      self._last = part_per_task * (self.pool.rank + 1)

      #Add the remainder to the last task
      if (Np % Nt) and (self.pool.rank == Nt - 1):
        self._last += Np % Nt

  def getPositions(self, first=None, last=None, save=True):

    #Get data pointer
    data = self.fp

    #Read in positions in Mpc/h
    if (first is None) or (last is None):
      positions = (data["0/Position"][:] + np.array(
          [0.5 / self.header["nc"] * self.header["box_size_mpch"], 
           0.5 / self.header["nc"] * self.header["box_size_mpch"],
           0],
          dtype=np.float32)) * self.Mpc_over_h
    else:
      positions = data["0/Position"][first:last] * self.Mpc_over_h

    #Enforce periodic boundary conditions
    for n in (0, 1):
      positions[:, n][positions[:, n] < 0] += self.header["box_size"]
      positions[:, n][
          positions[:, n] > self.header["box_size"]] -= self.header["box_size"]

    #Maybe save
    if save:
      self.positions = positions

    #Initialize useless attributes to None
    self.weights = None
    self.virial_radius = None
    self.concentration = None

    #Return
    return positions

def test_density_plane():
  """ Tests cutting density planes from snapshots against lenstools
  """
  klin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)

  cosmo = flowpm.cosmology.Planck15()

  a0 = 0.9
  r0 = flowpm.background.rad_comoving_distance(cosmo, a0)
  
  # Create a state vector
  initial_conditions = flowpm.linear_field(nc, bs, ipklin, batch_size=2)
  state = flowpm.lpt_init(cosmo, initial_conditions, a0)

  # Export the snapshot
  flowpm.io.save_state(cosmo, state, a0, 
           [nc, nc, nc], 
           [bs, bs, bs], 'snapshot_density_testing',
           attrs={'comoving_distance': r0})

  # Reload the snapshot with lenstools
  snapshot = FlowPMSnapshot.open('snapshot_density_testing')

  # Cut a lensplane in the middle of the volume
  lt_plane, resolution, NumPart = snapshot.cutPlaneGaussianGrid(normal=2,
                                                 plane_resolution=plane_resolution,   
                                                 center=(bs/2)*snapshot.Mpc_over_h,
                                                 thickness=(bs/4)*snapshot.Mpc_over_h,
                                                 left_corner=np.zeros(3)*snapshot.Mpc_over_h,
                                                 smooth=None,
                                                 kind='density')

  # Cut the same lensplane with flowpm
  fpm_plane = flowpm.raytracing.density_plane(state, nc, 
                                            center=nc/2, 
                                            width=nc/4,
                                            plane_resolution=plane_resolution)

  # Apply additional normalization terms to match lenstools definitions
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
  density_normalization = bs/4 * r0 / a0
  fpm_plane = fpm_plane * density_normalization * constant_factor

  # Checking first the mean value, which accounts for any normalization
  # issues
  assert_allclose(np.mean(fpm_plane[0]), np.mean(lt_plane), rtol=1e-5)

  # To check pixelwise difference, we need to do some smoothing as lenstools and
  # flowpm use different painting kernels
  smooth_lt_plane = np.fft.ifft2(fourier_gaussian(np.fft.fft2(lt_plane), 3)).real
  smooth_fpm_plane = np.fft.ifft2(fourier_gaussian(np.fft.fft2(fpm_plane[0]), 3)).real

  assert_allclose(smooth_fpm_plane, smooth_lt_plane, rtol=2e-2)

  

