import tensorflow as tf
import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import jax2tf
import haiku as hk
import sonnet as snt
import tree
import flowpm
import pickle
from flowpm.kernels import fftk, longrange_kernel, gradient_kernel, laplace_kernel
from flowpm.utils import cic_readout, compensate_cic, c2r3d, r2c3d
from flowpm.nn import NeuralSplineFourierFilter
from flowpm.cosmology import Planck15


def fun(x, a):
  network = NeuralSplineFourierFilter(n_knots=16, latent_size=32)
  return network(x, a)


fun = hk.without_apply_rng(hk.transform(fun))


def create_variable(path, value):
  name = '/'.join(map(str, path)).replace('~', '_')
  return tf.Variable(value, name=name)


class JaxNSFF(snt.Module):

  def __init__(self, params_filename, apply_fn, name=None):
    super().__init__(name=name)
    params = pickle.load(open(params_filename, "rb"))
    self._params = tree.map_structure_with_path(create_variable, params)
    self._apply = jax2tf.convert(lambda p, x, a: apply_fn(p, x, a))
    self._apply = tf.autograph.experimental.do_not_convert(self._apply)

  def __call__(self, input1, input2):
    return self._apply(self._params, input1, input2)


def make_neural_ode_fn(nc, batch_size, params_filename):
  net = JaxNSFF(params_filename, fun.apply)

  @tf.function(jit_compile=True)
  def apply_model(
      k,
      a,
  ):
    return net(k, a)

  def neural_nbody_ode(a, state, Omega_c, sigma8, Omega_b, n_s, h, w0):
    """
      Estimate force on the particles given a state.

      Parameters:
      -----------
      nc: int
        Number of cells in the field.
      
      batch_size: int
        Size of batches
        
      params_filename: 
        PM correction parameters 
        
      a : array_like or tf.TensorArray
        Scale factor
        
      state: tensor
        Input state tensor of shape (2, batch_size, npart, 3)

      Omega_c, sigma8, Omega_b, n_s,h, w0 : Scalar float Tensor
        Cosmological parameters

      Returns
      -------
      dpos: tensor (batch_size, npart, 3)
        Updated position at a given state
      dvel: tensor (batch_size, npart, 3)
        Updated velocity at a given state
      """

    pos = state[0]
    vel = state[1]
    kvec = fftk([nc, nc, nc], symmetric=False)
    cosmo = flowpm.cosmology.Planck15(
        Omega_c=Omega_c, sigma8=sigma8, Omega_b=Omega_b, n_s=n_s, h=h, w0=w0)
    delta = flowpm.cic_paint(tf.zeros([batch_size, nc, nc, nc]), pos)
    delta_k = r2c3d(delta)

    # Computes gravitational potential
    lap = tf.cast(laplace_kernel(kvec), tf.complex64)
    fknlrange = longrange_kernel(kvec, r_split=0)
    kweight = lap * fknlrange
    pot_k = tf.multiply(delta_k, kweight)

    # Apply a correction filter
    kk = tf.math.sqrt(sum((ki / np.pi)**2 for ki in kvec))
    pot_k = pot_k * tf.cast(
        (1. + apply_model(kk, tf.convert_to_tensor(a, tf.float32))),
        tf.complex64)

    # Computes gravitational forces

    forces = tf.stack([
        flowpm.cic_readout(
            c2r3d(tf.multiply(pot_k, gradient_kernel(kvec, i))), pos)
        for i in range(3)
    ],
                      axis=-1)
    forces = forces * 1.5 * cosmo.Omega_m

    #Computes the update of position (drift)
    dpos = 1. / (a**3 * flowpm.tfbackground.E(cosmo, a)) * vel

    #Computes the update of velocity (kick)
    dvel = 1. / (a**2 * flowpm.tfbackground.E(cosmo, a)) * forces

    return tf.stack([dpos, dvel], axis=0)

  return neural_nbody_ode
