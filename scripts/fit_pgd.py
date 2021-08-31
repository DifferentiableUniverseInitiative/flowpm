import numpy as np
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import flowpm
from flowpm import tfpm
from flowpm.utils import cic_paint
from flowpm.tfpower import linear_matter_power
from functools import partial
import flowpm.scipy.interpolate as interpolate

from nbodykit.cosmology import Cosmology
from nbodykit.cosmology.power.halofit import HalofitPower
from astropy.cosmology import Planck15
import astropy.units as u

flags.DEFINE_string("filename", "results_fit_PGD.pkl", "Output filename")
flags.DEFINE_integer("niter", 50, "Number of iterations of the first PGD fit")
flags.DEFINE_integer("niter_refine", 10, "Number of iterations of subsequent PGD fit")
flags.DEFINE_float("learning_rate", 0.1, "ADAM learning rate for the PGD optim")

flags.DEFINE_float("alpha0", 0.3, "Initial guess for alpha at z=0")
flags.DEFINE_float("kl0", 0.3, "Initial guess for kl at z=0")
flags.DEFINE_float("ks0", 10, "Initial guess for ks at z=0")

flags.DEFINE_boolean("fix_scales", True, "Whether to fix the scales after the initial fit at z=0")

flags.DEFINE_float("a_init", 0.1, "Initial scale factor")
flags.DEFINE_integer("nsteps", 40, "Number of steps in the N-body simulation")
flags.DEFINE_float("box_size", 64.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_integer("nc", 64,
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
  pdgized_state = state[0] + tfpm.PGD_correction(
      state, pgdparams, nc=[FLAGS.nc]*3, pm_nc_factor=FLAGS.B)
  # Step II: Painting
  field = cic_paint(tf.zeros([batch_size] + [FLAGS.nc] * 3), pdgized_state)[0]
  # Step III: Compute power spectrum
  k, pk = flowpm.power_spectrum(
      field,
      shape=field.shape,
      boxsize=np.array([FLAGS.box_size] * 3),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)
  # Step IV: compute loss
  weight = 1 - k / (np.pi * FLAGS.nc / FLAGS.box_size)
  weight = tf.convert_to_tensor(weight / weight.sum(), dtype=tf.float32)
  loss = tf.reduce_sum((weight * (1 - pk / target_pk))**2)
  
  if return_pk:
    return loss, pk
  else:
    return loss


def main(_):
  cosmology = flowpm.cosmology.Planck15()
  # Create a simple Planck15 cosmology without neutrinos, and makes sure sigma8
  # is matched
  nbdykit_cosmo = Cosmology.from_astropy(Planck15.clone(m_nu=0 * u.eV))
  nbdykit_cosmo = nbdykit_cosmo.match(sigma8=cosmology.sigma8.numpy())

  # Compute the k vectora that will be needed in the PGD fit
  k, _ = flowpm.power_spectrum(tf.zeros([1] + [FLAGS.nc] * 3),
      shape=tf.TensorShape([FLAGS.nc] * 3),
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
      batch_size=1)

  initial_state = flowpm.lpt_init(cosmology, initial_conditions, FLAGS.a_init)
  stages = np.linspace(FLAGS.a_init, 1., FLAGS.nsteps, endpoint=True)

  print('Starting simulation')
  # Run the Nbody
  states = flowpm.nbody(
      cosmology,
      initial_state,
      stages, [FLAGS.nc, FLAGS.nc, FLAGS.nc],
      pm_nc_factor=FLAGS.B,
      return_intermediate_states=True)
  print('Simulation done')

  # Initialize PGD params
  alpha = tf.Variable([FLAGS.alpha0],
                          dtype=tf.float32)
  scales = tf.Variable([FLAGS.kl0, FLAGS.ks0], dtype=tf.float32)
  pgdparams = tf.concat([alpha, scales], 0)
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

  # We begin by fitting the last time step
  for j, (a, state) in enumerate(states[::-1]):
    # Let's compute the target power spectrum at that scale factor
    target_pk = HalofitPower(nbdykit_cosmo, 1. / a - 1.)(k)

    for i in range(FLAGS.niter if j ==0 else FLAGS.niter_refine):
      optimizer.minimize(partial(pgd_loss, pgdparams, state, target_pk), 
                [alpha] if (FLAGS.fix_scales and j > 0) else [alpha, scales])

      if i % 10 == 0:
        loss, pk = pgd_loss(pgdparams, state, target_pk, return_pk=True)
        if i == 0:
          pk0 = pk
        print("step %d, loss:" % i, loss)
    print("Fitted params (alpha, kl, ks)", pgdparams)

    plt.loglog(k, target_pk, "k")
    plt.loglog(k, pk0 , ':', label='starting')
    plt.loglog(k, pk , '--', label='after n steps')
    plt.grid(which='both')
    plt.savefig('PGD_fit_%0.2f.png'%a)
    plt.close()

  # #Nyquist and number of modes in case we need weight for PGD fitting loss
  # kny = np.pi * FLAGS.nc / FLAGS.box_size
  # Nmodes = initpk(
  #     shape=initial_state.shape[1:],
  #     boxsize=np.array([FLAGS.box_size, FLAGS.box_size, FLAGS.box_size]),
  #     kmin=np.pi / FLAGS.box_size,
  #     dk=2 * np.pi / FLAGS.box_size)[1].numpy().real[1:-1]

  # ##Estimate PS for the last time snapshot
  # pk_array = []
  # for i in [ia]:
  #   print(i)
  #   final_field1 = flowpm.cic_paint(
  #       tf.zeros_like(initial_conditions), states2[i][1][0])
  #   final_field1 = tf.reshape(final_field1, [FLAGS.nc, FLAGS.nc, FLAGS.nc])
  #   k, power_spectrum1 = pkl(
  #       final_field1,
  #       shape=final_field1.shape,
  #       boxsize=np.array([FLAGS.box_size, FLAGS.box_size, FLAGS.box_size]),
  #       kmin=np.pi / FLAGS.box_size,
  #       dk=2 * np.pi / FLAGS.box_size)
  #   pk_array.append(power_spectrum1)
  # pk_jax = []
  # cosmo = jc.Planck15()

  # for i in [ia]:
  #   print(i)
  #   pk_jax.append(power.nonlinear_matter_power(cosmo, k, states2[i][0]))

  # print(len(k))
  # print(len(pk_jax[-1]))
  # print('PS computed')
  # #Finally fit
  # #Loss
  # weight = 1 - k / kny
  # weight = weight / weight.sum()
  # weight = tf.cast(weight, tf.float32)
  # kl, ks = 0.3, 12
  # alpha = 0.3
  # params0 = np.array([alpha, kl, ks]).astype(np.float32)
  # params = tf.Variable(
  #     name='pgd',
  #     shape=(3),
  #     dtype=tf.float32,
  #     initial_value=params0,
  #     trainable=True)
  # print('Init params : ', params)

  # opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  # losses, pparams = [], []
  # print(params0)
  # for i in range(niters):
  #   loss, grads = get_grads(params, states2, initial_conditions, pk_jax, weight)
  #   opt.apply_gradients(zip([grads], [params]))
  #   print(i, loss, params)
  #   losses.append(loss)
  #   pparams.append(params.numpy().copy())

  # print('Final params : ', params)
  # #Plot fit params
  # plt.plot(losses)
  # plt.semilogy()
  # plt.grid(which='both')
  # plt.savefig('losses.png')
  # plt.close()

  # k, ps, field = pgd(params, states2, initial_conditions)
  # k, ps0, field0 = pgd(tf.constant(params0), states2, initial_conditions)
  # plt.plot(k, pk_array[ia] / pk_jax[ia], "k:", label='B=1')
  # plt.plot(k, ps0 / pk_jax[ia], 'k--', label='0')
  # plt.semilogx()
  # plt.grid(which='both')
  # for i in range(niters // 10):
  #   j = 10 * i + 9
  #   try:
  #     k, ps, field = pgd(tf.constant(pparams[j]), states2, initial_conditions)
  #     plt.plot(k, ps / pk_jax[ia], '-', label=j)
  #   except:
  #     pass
  # plt.legend()
  # plt.savefig('pgdfit.png')
  # pickle.dump(
  #     {
  #         'B': FLAGS.B,
  #         'nsteps': FLAGS.nsteps,
  #         'pgdparams': params.numpy(),
  #         'alpha': params.numpy()[0],
  #         'kl': params.numpy()[1],
  #         'ks': params.numpy()[2],
  #     }, open(FLAGS.filename, "wb"))


if __name__ == "__main__":
  app.run(main)
