import numpy as np
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import flowpm
from flowpm import tfpm
from flowpm.utils import cic_paint, compensate_cic
from flowpm.tfpower import linear_matter_power
from functools import partial
import flowpm.scipy.interpolate as interpolate

from nbodykit.cosmology import Cosmology
from nbodykit.cosmology.power.halofit import HalofitPower
from astropy.cosmology import Planck15
import astropy.units as u

flags.DEFINE_string("filename", "results_fit_PGD.pkl", "Output filename")
flags.DEFINE_integer("niter", 51, "Number of iterations of the first PGD fit")
flags.DEFINE_integer("niter_refine", 51,
                     "Number of iterations of subsequent PGD fit")
flags.DEFINE_float("learning_rate", 0.1, "ADAM learning rate for the PGD optim")
flags.DEFINE_integer("batch_size", 2, "Number of random draws")

flags.DEFINE_float("alpha0", 0.1, "Initial guess for alpha at z=0")
flags.DEFINE_float("kl0", 0.3, "Initial guess for kl at z=0")
flags.DEFINE_float("ks0", 10, "Initial guess for ks at z=0")

flags.DEFINE_float("a_init", 0.1, "Initial scale factor")
flags.DEFINE_integer("nsteps", 15, "Number of steps in the N-body simulation")
flags.DEFINE_float("box_size", 205.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_integer("nc", 128,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_integer("B", 1, "Scale resolution factor")

FLAGS = flags.FLAGS


def pgd_loss(pgdparams, state, target_pk, return_pk=False):
  """
  Defines the loss function for the PGD parameters
  """
  shape = state.get_shape()
  batch_size = shape[1]

  # Step I: Apply PGD to the state vector
  pdgized_state = tfpm.pgd(
      state, pgdparams, nc=[FLAGS.nc] * 3, pm_nc_factor=FLAGS.B)

  # Step II: Painting and compensating for cic window
  field = cic_paint(tf.zeros([batch_size] + [FLAGS.nc] * 3), pdgized_state[0])
  field = compensate_cic(field)

  # Step III: Compute power spectrum
  k, pk = flowpm.power_spectrum(
      field,
      boxsize=np.array([FLAGS.box_size] * 3),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)
  # Averaging pk over realisations
  pk = tf.reduce_mean(pk, axis=0)

  # Step IV: compute loss
  loss = tf.reduce_sum((1. - pk / target_pk)**2)
  if return_pk:
    return loss, pk
  else:
    return loss


def fit_nbody(cosmo, state, stages, nc, pm_nc_factor=1, name="NBody"):
  """
  Integrate the evolution of the state across the givent stages
  Parameters:
  -----------
  cosmo: cosmology
    Cosmological parameter object
  state: tensor (3, batch_size, npart, 3)
    Input state
  stages: array
    Array of scale factors
  nc: int, or list of ints
    Number of cells
  pm_nc_factor: int
    Upsampling factor for computing
  pgdparams: array
    list of pgdparameters [alpha, kl, ks] of size len(stages) - 1
  Returns
  -------
  state: tensor (3, batch_size, npart, 3), or list of states
    Integrated state to final condition, or list of intermediate steps
  """
  with tf.name_scope(name):
    state = tf.convert_to_tensor(state, name="state")

    # Create a simple Planck15 cosmology without neutrinos, and makes sure sigma8
    # is matched
    nbdykit_cosmo = Cosmology.from_astropy(Planck15.clone(m_nu=0 * u.eV))
    nbdykit_cosmo = nbdykit_cosmo.match(sigma8=cosmo.sigma8.numpy())

    if isinstance(nc, int):
      nc = [nc, nc, nc]

    # Unrolling leapfrog integration to make tf Autograph happy
    if len(stages) == 0:
      return state

    ai = stages[0]

    # first force calculation for jump starting
    state = tfpm.force(cosmo, state, nc, pm_nc_factor=pm_nc_factor)

    k, _ = flowpm.power_spectrum(
        tf.zeros([1] + [FLAGS.nc] * 3),
        boxsize=np.array([FLAGS.box_size] * 3),
        kmin=np.pi / FLAGS.box_size,
        dk=2 * np.pi / FLAGS.box_size)

    params = tf.Variable([FLAGS.alpha0, FLAGS.kl0, FLAGS.ks0], dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    x, p, f = ai, ai, ai
    pgdparams = []
    scale_factors = []
    # Loop through the stages
    for i in range(len(stages) - 1):
      a0 = stages[i]
      a1 = stages[i + 1]
      ah = (a0 * a1)**0.5

      # Kick step
      state = tfpm.kick(cosmo, state, p, f, ah)
      p = ah

      # Drift step
      state = tfpm.drift(cosmo, state, x, p, a1)

      # Let's compute the target power spectrum at that scale factor
      target_pk = HalofitPower(nbdykit_cosmo, 1. / a1 - 1.)(k).astype('float32')

      for j in range(FLAGS.niter if i == 0 else FLAGS.niter_refine):
        optimizer.minimize(partial(pgd_loss, params, state, target_pk), params)

        if j % 10 == 0:
          loss, pk = pgd_loss(params, state, target_pk, return_pk=True)
          if j == 0:
            pk0 = pk
          print("step %d, loss:" % j, loss)
      pgdparams.append(params.numpy())
      scale_factors.append(a1)
      print("Sim step %d, fitted params (alpha, kl, ks)" % i, pgdparams[-1])
      plt.loglog(k, target_pk, "k")
      plt.loglog(k, pk0, ':', label='starting')
      plt.loglog(k, pk, '--', label='after n steps')
      plt.grid(which='both')
      plt.savefig('PGD_fit_%0.2f.png' % a1)
      plt.close()
      # Optional PGD correction step
      state = tfpm.pgd(state, params, nc, pm_nc_factor=pm_nc_factor)
      x = a1

      # Force
      state = tfpm.force(cosmo, state, nc, pm_nc_factor=pm_nc_factor)
      f = a1

      # Kick again
      state = tfpm.kick(cosmo, state, p, f, a1)
      p = a1

    return state, scale_factors, pgdparams


def main(_):
  cosmology = flowpm.cosmology.Planck15()

  # Compute the k vectora that will be needed in the PGD fit
  k, _ = flowpm.power_spectrum(
      tf.zeros([1] + [FLAGS.nc] * 3),
      boxsize=np.array([FLAGS.box_size] * 3),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)

  # Create some initial conditions
  klin = tf.constant(np.logspace(-4, 1, 512), dtype=tf.float32)
  pk = linear_matter_power(cosmology, klin)
  pk_fun = lambda x: tf.cast(
      tf.reshape(
          interpolate.interp_tf(
              tf.reshape(tf.cast(x, tf.float32), [-1]), klin, pk), x.shape), tf.
      complex64)

  initial_conditions = flowpm.linear_field(
      [FLAGS.nc, FLAGS.nc, FLAGS.nc],
      [FLAGS.box_size, FLAGS.box_size, FLAGS.box_size],
      pk_fun,
      batch_size=FLAGS.batch_size)

  initial_state = flowpm.lpt_init(cosmology, initial_conditions, FLAGS.a_init)
  stages = np.linspace(FLAGS.a_init, 1., FLAGS.nsteps, endpoint=True)

  print('Starting simulation')
  state, scale_factors, pgdparams = fit_nbody(
      cosmology,
      initial_state,
      stages, [FLAGS.nc, FLAGS.nc, FLAGS.nc],
      pm_nc_factor=FLAGS.B)
  print('Simulation done')

  pickle.dump(
      {
          'B': FLAGS.B,
          'nsteps': FLAGS.nsteps,
          'params': pgdparams,
          'scale_factors': scale_factors,
          'cosmology': cosmology.to_dict(),
          'boxsize': FLAGS.box_size,
          'nc': FLAGS.nc
      }, open(FLAGS.filename, "wb"))


if __name__ == "__main__":
  app.run(main)
