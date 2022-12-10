import tensorflow as tf
import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import jax2tf
import haiku as hk
import sonnet as snt
import tree
import pickle
from flowpm.nn import NeuralSplineFourierFilter

def fun(x, a):
  network = NeuralSplineFourierFilter(n_knots=16, latent_size=32)
  return network(x, a)


fun = hk.without_apply_rng(hk.transform(fun))
params = pickle.load( open( "/local/home/dl264294/flowpm/notebooks/camels_25_64_pkloss.params", "rb" ) )

def create_variable(path, value):
  name = '/'.join(map(str, path)).replace('~', '_')
  return tf.Variable(value, name=name)


class JaxNSFF(snt.Module):

  def __init__(self, params, apply_fn, name=None):
    super().__init__(name=name)
    self._params = tree.map_structure_with_path(create_variable, params)
    self._apply = jax2tf.convert(lambda p, x, a: apply_fn(p, x, a))
    self._apply = tf.autograph.experimental.do_not_convert(self._apply)

  def __call__(self, input1, input2):
    return self._apply(self._params, input1, input2)

net = JaxNSFF(params, fun.apply)


@tf.function(autograph=False, input_signature=[tf.TensorSpec([128,128,128]),
                                               tf.TensorSpec([]),])
def forward(x,a):
  return net(x,a)

to_save = tf.Module()
to_save.forward = forward
to_save.params = list(net.variables)
tf.saved_model.save(to_save, "/local/home/dl264294/flowpm/saved_model")
