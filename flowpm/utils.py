# Module storing a few tensorflow function to implement FastPM
import numpy as np
import numpy
import tensorflow as tf


def cic_readout(mesh, part, cube_size=None, boxsize=None):
  """
      - mesh is a cube
      - part is a list of particles (:, 3), positions assumed to be in
  mesh units if boxsize is None
      - cube_size is the size of the cube in mesh units
  """

  if cube_size is None: cube_size = int(mesh.shape[0].value)
  if boxsize is not None:
      part = tf.multiply(part, cube_size/boxsize)

  # Extract the indices of all the mesh points affected by each particles
  part = tf.expand_dims(part, 1)
  floor = tf.floor(part)
  connection = tf.constant([[[0, 0, 0], [1., 0, 0],[0., 1, 0],[0., 0, 1],[1., 1, 0],[1., 0, 1],[0., 1, 1],[1., 1, 1]]])
  neighboor_coords = tf.add(floor, connection)

  kernel = 1. - tf.abs(part - neighboor_coords)
  kernel = tf.reduce_prod(kernel, axis=-1, keepdims=False)

#     if cube_size is not None:
  neighboor_coords = tf.cast(neighboor_coords, tf.int32)
  neighboor_coords = tf.mod(neighboor_coords , cube_size)

  meshvals = tf.gather_nd(mesh, neighboor_coords)
  weightedvals = tf.multiply(meshvals, kernel)
  value = tf.reduce_sum(weightedvals, axis=1)
  return value

def r2c3d(rfield, norm=None, dtype=tf.complex64, name=None):
    if norm is None: norm = tf.cast(tf.reduce_prod(tf.shape(rfield)), dtype)
    else: norm = tf.cast(norm, dtype)
    cfield = tf.multiply(tf.spectral.fft3d(tf.cast(rfield, dtype)), 1/norm, name=name)
    return cfield

def c2r3d(cfield, norm=None, dtype=tf.float32, name=None):
    if norm is None: norm = tf.cast(tf.reduce_prod(tf.shape(cfield)), dtype)
    else: norm = tf.cast(norm, dtype)
    rfield = tf.multiply(tf.cast(tf.spectral.ifft3d(cfield), dtype), norm, name=name)
    return rfield

def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
    """ return k_vector given a shape (nc, nc, nc) and boxsize
    """
    k = []
    for d in range(len(shape)):
        kd = numpy.fft.fftfreq(shape[d])
        kd *= 2 * numpy.pi / boxsize * shape[d]
        kdshape = numpy.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)

        k.append(kd.astype(dtype))
    del kd, kdshape
    return k


def laplace(config):
    kvec = config['kvec']
    kk = sum(ki**2 for ki in kvec)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    wts = 1/kk
    imask = (~(kk==0)).astype(int)
    wts *= imask
    return wts



def gradient(config, dir):
    kvec = config['kvec']
    bs, nc = config['boxsize'], config['nc']
    cellsize = bs/nc
    w = kvec[dir] * cellsize
    a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
    wts = a*1j
    return wts



def kernellongrange(config, r_split):
    if r_split != 0:
        kk = sum(ki ** 2 for ki in config['kvec'])
        return numpy.exp(-kk * r_split**2)
    else:
        return 1.


def longrange(config, x, delta_k, split=0, factor=1):
    """ like long range, but x is a list of positions """
    # use the four point kernel to suppresse artificial growth of noise like terms

    ndim = 3
    norm = config['nc']**3
    lap = laplace(config)
    fknlrange = kernellongrange(config, split)
    kweight = lap * fknlrange
    pot_k = tf.multiply(delta_k, kweight)

    f = []
    for d in range(ndim):
        force_dc = tf.multiply(pot_k, gradient(config, d))
        #forced = tf.multiply(tf.spectral.irfft3d(force_dc), config['nc']**3)
        forced = c2r3d(force_dc, norm=norm)
        force = cic_readout(forced, x)
        f.append(force)

    f = tf.stack(f, axis=1)
    f = tf.multiply(f, factor)
    return f
