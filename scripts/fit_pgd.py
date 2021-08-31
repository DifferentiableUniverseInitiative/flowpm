import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

sys.path.append('../../flowpm/')
import pickle
import flowpm
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from flowpm.tfpower import linear_matter_power
from absl import app
from absl import flags
from flowpm import tfpm
import jax_cosmo as jc
from flowpm.pk import pk as pkl
from flowpm.pk import _initialize_pk as initpk
import jax_cosmo.power as power
from flowpm.utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout

flags.DEFINE_string("filename", "results_fit_PGD.pkl", "Output filename")
flags.DEFINE_float("box_size", 64.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_integer("nc", 64,
                     "Number of transverse voxels in the simulation volume")
flags.DEFINE_integer("nsteps", 40, "Number of steps in the N-body simulation")
flags.DEFINE_integer("B", 1, "Scale resolution factor")

FLAGS = flags.FLAGS


#######################################
#Same correction function as flowpm but writing as tf.function
@tf.function
def PGD_correction(state, pgdparams):
  """                                                                                                                                                                                                                                                                                                                          """
  print('new graph')
  state = tf.convert_to_tensor(state, name="state")
  pm_nc_factor = FLAGS.B
  shape = state.get_shape()
  batch_size = shape[1]
  ncpm = [FLAGS.nc, FLAGS.nc, FLAGS.nc]
  ncf = [FLAGS.nc * pm_nc_factor] * 3
  rho = tf.zeros([batch_size] + ncf)
  wts = tf.ones((batch_size, ncpm[0] * ncpm[1] * ncpm[2]))
  nbar = ncpm[0] * ncpm[1] * ncpm[2] / (ncf[0] * ncf[1] * ncf[2])

  rho = cic_paint(rho, tf.multiply(state[0], pm_nc_factor), wts)
  rho = tf.multiply(rho, 1. / nbar)
  delta_k = r2c3d(rho, norm=ncf[0] * ncf[1] * ncf[2])
  alpha, kl, ks = tf.split(pgdparams, 3)
  update = tfpm.apply_PGD(
      tf.multiply(state[0], pm_nc_factor),
      delta_k,
      alpha,
      kl,
      ks,
  )
  update = tf.expand_dims(update, axis=0) / pm_nc_factor
  return update


@tf.function
def pgd(params, states, initial_conditions):
  print('pgd graph')
  ia = -1
  dx = PGD_correction(states[ia][1], params)
  new_state = states[ia][1][0] + dx
  final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), new_state)
  final_field = tf.reshape(final_field, [FLAGS.nc, FLAGS.nc, FLAGS.nc])
  k, power_spectrum = pkl(
      final_field,
      shape=final_field.shape,
      boxsize=np.array([FLAGS.box_size, FLAGS.box_size, FLAGS.box_size]),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)

  return k, power_spectrum, final_field


#######################################
@tf.function
def get_grads(params, states, initial_conditions, pk_jax, weight):
  ia = -1
  with tf.GradientTape() as tape:
    tape.watch(params)
    k, ps, field = pgd(params, states, initial_conditions)
    psref = tf.convert_to_tensor(pk_jax[ia])
    loss = tf.reduce_sum((weight * (1 - ps / psref))**2)
  grad = tape.gradient(loss, params)
  return loss, grad


def main(_):

  cosmology = flowpm.cosmology.Planck15()
  fftk = flowpm.kernels.fftk
  laplace_kernel = flowpm.kernels.laplace_kernel
  gradient_kernel = flowpm.kernels.gradient_kernel
  ia = -1  #Scale factor to fit for
  niters = 50
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
  initial_state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)
  stages = np.linspace(0.1, 1., FLAGS.nsteps, endpoint=True)
  # Run the Nbody
  states2 = flowpm.nbody(
      cosmology,
      initial_state,
      stages, [FLAGS.nc, FLAGS.nc, FLAGS.nc],
      pm_nc_factor=FLAGS.B,
      return_intermediate_states=True)
  print('Simulation done')
  #Nyquist and number of modes in case we need weight for PGD fitting loss
  kny = np.pi * FLAGS.nc / FLAGS.box_size
  Nmodes = initpk(
      shape=initial_state.shape[1:],
      boxsize=np.array([FLAGS.box_size, FLAGS.box_size, FLAGS.box_size]),
      kmin=np.pi / FLAGS.box_size,
      dk=2 * np.pi / FLAGS.box_size)[1].numpy().real[1:-1]

  ##Estimate PS for the last time snapshot
  pk_array = []
  for i in [ia]:
    print(i)
    final_field1 = flowpm.cic_paint(
        tf.zeros_like(initial_conditions), states2[i][1][0])
    final_field1 = tf.reshape(final_field1, [FLAGS.nc, FLAGS.nc, FLAGS.nc])
    k, power_spectrum1 = pkl(
        final_field1,
        shape=final_field1.shape,
        boxsize=np.array([FLAGS.box_size, FLAGS.box_size, FLAGS.box_size]),
        kmin=np.pi / FLAGS.box_size,
        dk=2 * np.pi / FLAGS.box_size)
    pk_array.append(power_spectrum1)
  pk_jax = []
  cosmo = jc.Planck15()

  for i in [ia]:
    print(i)
    pk_jax.append(power.nonlinear_matter_power(cosmo, k, states2[i][0]))

  print(len(k))
  print(len(pk_jax[-1]))
  print('PS computed')
  #Finally fit
  #Loss
  weight = 1 - k / kny
  weight = weight / weight.sum()
  weight = tf.cast(weight, tf.float32)
  kl, ks = 0.3, 12
  alpha = 0.3
  params0 = np.array([alpha, kl, ks]).astype(np.float32)
  params = tf.Variable(
      name='pgd',
      shape=(3),
      dtype=tf.float32,
      initial_value=params0,
      trainable=True)
  print('Init params : ', params)

  opt = tf.keras.optimizers.Adam(learning_rate=0.1)
  losses, pparams = [], []
  print(params0)
  for i in range(niters):
    loss, grads = get_grads(params, states2, initial_conditions, pk_jax, weight)
    opt.apply_gradients(zip([grads], [params]))
    print(i, loss, params)
    losses.append(loss)
    pparams.append(params.numpy().copy())

  print('Final params : ', params)
  #Plot fit params
  plt.plot(losses)
  plt.semilogy()
  plt.grid(which='both')
  plt.savefig('losses.png')
  plt.close()

  k, ps, field = pgd(params, states2, initial_conditions)
  k, ps0, field0 = pgd(tf.constant(params0), states2, initial_conditions)
  plt.plot(k, pk_array[ia] / pk_jax[ia], "k:", label='B=1')
  plt.plot(k, ps0 / pk_jax[ia], 'k--', label='0')
  plt.semilogx()
  plt.grid(which='both')
  for i in range(niters // 10):
    j = 10 * i + 9
    try:
      k, ps, field = pgd(tf.constant(pparams[j]), states2, initial_conditions)
      plt.plot(k, ps / pk_jax[ia], '-', label=j)
    except:
      pass
  plt.legend()
  plt.savefig('pgdfit.png')
  pickle.dump(
      {
          'B': FLAGS.B,
          'nsteps': FLAGS.nsteps,
          'pgdparams': params.numpy(),
          'alpha': params.numpy()[0],
          'kl': params.numpy()[1],
          'ks': params.numpy()[2],
      }, open(FLAGS.filename, "wb"))


if __name__ == "__main__":
  app.run(main)
