# Testing raytracing functions against lenstools
import tensorflow as tf
import numpy as np
import flowpm
import lenstools as lt
import bigfile


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
    #Translate fastPM header into lenstools header#
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
    header["redshift"] = a2z(bf_header["Time"][0])
    header["comoving_distance"] = bf_header["comoving_distance"][0] * 1.0e3
    header["scale_factor"] = bf_header["Time"][0]

    #Box size in kpc/h
    header["box_size"] = bf_header["BoxSize"][0] * 1.0e3
    #Plane Resolution
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
          [0.5 / plane_size * box_size, 0.5 / plane_size * box_size, 0],
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
