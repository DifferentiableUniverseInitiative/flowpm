import tensorflow as tf
import numpy as np
from astropy.cosmology import Planck15
from .utils import *

class ParticleMeshSimulation():

  def __init__(self,
               box_size=100.,
               cosmology=Planck15,
               stages=np.linspace(0.1, 1.0, 5, endpoint=True),
               dtype=np.float32):
    """
    Initializes a PM simulation

    Parameters:
    -----------
    box_size: float
      Size of the simulation box (Mpc/h) TODO: check units
    cosmology: astropy.cosmology
      Cosmology object to use for the simulation
    stages: array
      Stages of the simulation
    """
    self.box_size = box_size
    self.cosmo = cosmology
    self.stepping = leapfrog(stages)
    super(ParticleMeshLayer, self).__init__(*args, **kwargs)


  def call(self, state):
    """
    Applies evolution to the input state

    Parameters:
    -----------
    state: tensor (batch_size, nc, nc, nc)
    """
    # Stacks the building blocks required
    for action, ai, ac, af in self.stepping:
      state = action(state, ai, ac, af)
