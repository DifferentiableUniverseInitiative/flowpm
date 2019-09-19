import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose

from flowpm.tfpm import cic_paint
from pmesh.pm import ParticleMesh

def test_cic_paint():
  bs = 50
  nc = 16
  pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
  nparticle = 100
  pos = bs*np.random.random(3*nparticle).reshape(-1, 3).astype(np.float32)
  wts = np.random.random(nparticle).astype(np.float32)

  # Painting with pmesg
  pmmesh = pm.paint(pos, mass=wts)

  with tf.Session() as sess:
      mesh = cic_paint(tf.zeros((1, nc, nc, nc), dtype=tf.float32),
                       (pos*nc/bs).reshape((1, nparticle, 3)),
                       weight=wts.reshape(1, nparticle))
      sess.run(tf.global_variables_initializer())
      tfmesh = sess.run(mesh)

  assert_allclose(pmmesh, tfmesh[0], atol=1e-06)
