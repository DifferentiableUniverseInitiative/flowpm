from bigfile import File
import tensorflow as tf


def save_state(cosmo, state, a, nc, boxsize, filename, attrs={}):
  """
  Saves a state vector from a given simulation snapshot.

  Note: This is not exactly fastpm compatible.

  Parameters
  ----------
  cosmo: cosmology object
    Comology used to create the snapshot

  state: Tensor, nbody state
    State tensor with particle position and speed

  a: float
    Scale factor of the current state

  nc: list of int
    Number of cells

  boxsize: list of float
    Length of the simulation volume

  filename: str
    Export file name
  """
  with File(filename, create=True) as ff:
    with ff.create('Header') as bb:
      bb.attrs['NC'] = nc
      bb.attrs['BoxSize'] = boxsize
      bb.attrs['OmegaCDM'] = cosmo.Omega_c.numpy()
      bb.attrs['OmegaB'] = cosmo.Omega_b.numpy()
      bb.attrs['OmegaK'] = cosmo.Omega_k.numpy()
      bb.attrs['h'] = cosmo.h.numpy()
      bb.attrs['Sigma8'] = cosmo.sigma8.numpy()
      bb.attrs['w0'] = cosmo.w0.numpy()
      bb.attrs['wa'] = cosmo.wa.numpy()
      bb.attrs['Time'] = a
      bb.attrs['M0'] = [1.]  # Hummmm I don't know about this one
      for key in attrs:
        try:
          # best effort
          bb.attrs[key] = attrs[key]
        except:
          pass
    # Factor to convert speed and position back to Mpc/h
    scaling_factor = 1. / tf.convert_to_tensor(nc, dtype=tf.float32)
    scaling_factor = scaling_factor * tf.convert_to_tensor(
        boxsize, dtype=tf.float32)
    scaling_factor = tf.reshape(scaling_factor, [1, 3])
    # Export each batch entry as its own block
    for i in range(state.shape[1]):
      ff.create_from_array('%d/Position' % i,
                           (state[0, i] * scaling_factor).numpy())
      ff.create_from_array('%d/Velocity' % i,
                           (state[1, i] * scaling_factor).numpy())
