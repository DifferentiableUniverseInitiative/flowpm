import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

from flowpm.tfpm import linear_field

np.random.seed(0)

def test_linear_field_shape():
  bs = 50
  nc = 16

  klin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)

  with tf.Session() as sess:
    field = linear_field(nc, bs, ipklin, batch_size=5)
    sess.run(tf.global_variables_initializer())
    tfread = sess.run(field)

  assert tfread.shape == (5, 16, 16, 16)
