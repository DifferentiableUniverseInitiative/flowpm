from astropy.cosmology import Planck15
import tensorflow as tf

def cic_paint(mesh, part, weight=None, name=None):
  """
  Paints particules on a 3D mesh.

  Parameters:
  -----------
  mesh: tensor (batch_size, nc, nc, nc)
    Input 3D mesh tensor

  part: tensor (batch_size, npart, 3)
    List of 3D particle coordinates, assumed to be in mesh units if
    boxsize is None

  weight: tensor (batch_size, npart)
    List of weights  for each particle
  """
  with tf.name_scope(name, "CICpaint", [mesh, part, weight]):
    shape = tf.shape(mesh)
    batch_size, nc = shape[0], shape[1]

    # Extract the indices of all the mesh points affected by each particles
    part = tf.expand_dims(part, 2)
    floor = tf.floor(part)
    connection = tf.expand_dims(tf.constant([[[0, 0, 0], [1., 0, 0],[0., 1, 0],
                                              [0., 0, 1],[1., 1, 0],[1., 0, 1],
                                              [0., 1, 1],[1., 1, 1]]]), 0)

    neighboor_coords = floor + connection
    kernel = 1. - tf.abs(part - neighboor_coords)
    # Replacing the reduce_prod op by manual multiplication
    # TODO: figure out why reduce_prod was crashing the Hessian computation
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    if weight is not None: kernel = tf.multiply(tf.expand_dims(weight, axis=-1) , kernel)

    neighboor_coords = tf.cast(neighboor_coords, tf.int32)
    neighboor_coords = tf.mod(neighboor_coords , nc)
    print(neighboor_coords)
    print(kernel)
    # Adding batch dimension to the neighboor coordinates
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
    b = tf.tile(batch_idx, [1] + list(neighboor_coords.get_shape()[1:-1]) + [1])
    neighboor_coords = tf.concat([b, neighboor_coords], axis=-1)

    update = tf.scatter_nd(tf.reshape(neighboor_coords, (-1, 8,4)),
                           tf.reshape(kernel, (-1, 8)),
                           [batch_size, nc, nc, nc])
    mesh = mesh + update
    return mesh

def Force(state, box_size, cosmology=Planck15, pm_nc_factor=1, dtype=tf.float32):
  """
  Estimate force on the particles given a state.

  Parameters:
  -----------
  state: tensor
    Input state tensor of shape (batch_size, nc, nc, nc)

  box_size: float
    Size of the simulation volume (Mpc/h) TODO: check units

  cosmology: astropy.cosmology
    Cosmology object

  pm_nc_factor: int
    TODO: @modichirag please add doc
  """
  rho = tf.zeros((ncf, ncf, ncf))
  wts = tf.ones(nc**3)
  nbar = nc**3/ncf**3

  rho = cic_paint(rho, tf.multiply(state[0], ncf/bs), wts)
  rho = tf.multiply(rho, 1/nbar)  ###I am not sure why this is not needed here
  delta_k = r2c3d(rho, norm=ncf**3)
  fac = dtype(1.5 * config['cosmology'].Om0)
  update = longrange(config['f_config'], tf.multiply(state[0], ncf/bs), delta_k, split=0, factor=fac)

  update = tf.expand_dims(update, axis=0)

  indices = tf.constant([[2]])
  shape = state.shape
  update = tf.scatter_nd(indices, update, shape)
  mask = tf.stack((tf.ones_like(state[0]), tf.ones_like(state[0]), tf.zeros_like(state[0])), axis=0)
  state = tf.multiply(state, mask)
  state = tf.add(state, update)
  return state
