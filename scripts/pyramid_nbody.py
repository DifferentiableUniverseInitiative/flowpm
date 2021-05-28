from mpi4py import MPI
comm = MPI.COMM_WORLD

import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf
from mesh_tensorflow.hvd_simd_mesh_impl import HvdSimdMeshImpl
import flowpm.mesh_ops as mpm
import flowpm.mtfpm as mtfpm
import flowpm.mesh_utils as mesh_utils
import flowpm
from astropy.cosmology import Planck15
##

cosmology = Planck15
tf.random.set_random_seed(200*comm.Get_rank())

tf.flags.DEFINE_integer("nc", 128, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 512, "Box Size [Mpc/h]")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 3, "Number of time steps")

#pyramid flags
tf.flags.DEFINE_integer("dsample", 2, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 32, "halo size")

#mesh flags
tf.flags.DEFINE_integer("nx", 1, "# blocks along x")
tf.flags.DEFINE_integer("ny", 1, "# blocks along y")

FLAGS = tf.flags.FLAGS

def nbody_fn(mesh,
             klin, plin,
             nc=FLAGS.nc,
             bs=FLAGS.box_size,
             batch_size=FLAGS.batch_size,
             a0=FLAGS.a0,
             a=FLAGS.af,
             nsteps=FLAGS.nsteps,
             dtype=tf.float32):
  """ Pyramid N-body function
  """
  stages = np.linspace(a0, a, nsteps, endpoint=True)

  # Define the named dimensions
  # Parameters of the small scales decomposition
  n_block_x = FLAGS.nx
  n_block_y = FLAGS.ny
  n_block_z = 1
  halo_size = FLAGS.hsize

  # Parameters of the large scales decomposition
  downsampling_factor = FLAGS.dsample
  lnc = nc // 2**downsampling_factor

  fx_dim = mtf.Dimension("nx", nc)
  fy_dim = mtf.Dimension("ny", nc)
  fz_dim = mtf.Dimension("nz", nc)

  tfx_dim = mtf.Dimension("tx", nc)
  tfy_dim = mtf.Dimension("ty", nc)
  tfz_dim = mtf.Dimension("tz", nc)

  # Dimensions of the low resolution grid
  x_dim = mtf.Dimension("nx_lr", lnc)
  y_dim = mtf.Dimension("ny_lr", lnc)
  z_dim = mtf.Dimension("nz_lr", lnc)

  tx_dim = mtf.Dimension("tx_lr", lnc)
  ty_dim = mtf.Dimension("ty_lr", lnc)
  tz_dim = mtf.Dimension("tz_lr", lnc)

  nx_dim = mtf.Dimension('nx_block', n_block_x)
  ny_dim = mtf.Dimension('ny_block', n_block_y)
  nz_dim = mtf.Dimension('nz_block', n_block_z)

  sx_dim = mtf.Dimension('sx_block', nc // n_block_x)
  sy_dim = mtf.Dimension('sy_block', nc // n_block_y)
  sz_dim = mtf.Dimension('sz_block', nc // n_block_z)

  batch_dim = mtf.Dimension("batch", batch_size)
  pk_dim = mtf.Dimension("npk", len(plin))
  pk = mtf.import_tf_tensor(mesh, plin.astype('float32'), shape=[pk_dim])

  # Compute necessary Fourier kernels
  kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False)
  kx = mtf.import_tf_tensor(mesh,
                            kvec[0].squeeze().astype('float32'),
                            shape=[tfx_dim])
  ky = mtf.import_tf_tensor(mesh,
                            kvec[1].squeeze().astype('float32'),
                            shape=[tfy_dim])
  kz = mtf.import_tf_tensor(mesh,
                            kvec[2].squeeze().astype('float32'),
                            shape=[tfz_dim])
  kv = [ky, kz, kx]

  # kvec for low resolution grid
  kvec_lr = flowpm.kernels.fftk([lnc, lnc, lnc], symmetric=False)

  kx_lr = mtf.import_tf_tensor(mesh,
                               kvec_lr[0].squeeze().astype('float32') /
                               2**downsampling_factor,
                               shape=[tx_dim])
  ky_lr = mtf.import_tf_tensor(mesh,
                               kvec_lr[1].squeeze().astype('float32') /
                               2**downsampling_factor,
                               shape=[ty_dim])
  kz_lr = mtf.import_tf_tensor(mesh,
                               kvec_lr[2].squeeze().astype('float32') /
                               2**downsampling_factor,
                               shape=[tz_dim])
  kv_lr = [ky_lr, kz_lr, kx_lr]

  # kvec for high resolution blocks
  padded_sx_dim = mtf.Dimension('padded_sx_block',
                                nc // n_block_x + 2 * halo_size)
  padded_sy_dim = mtf.Dimension('padded_sy_block',
                                nc // n_block_y + 2 * halo_size)
  padded_sz_dim = mtf.Dimension('padded_sz_block',
                                nc // n_block_z + 2 * halo_size)
  kvec_hr = flowpm.kernels.fftk([
      nc // n_block_x + 2 * halo_size, nc // n_block_y + 2 * halo_size,
      nc // n_block_z + 2 * halo_size
  ],
                                symmetric=False)

  kx_hr = mtf.import_tf_tensor(mesh,
                               kvec_hr[0].squeeze().astype('float32'),
                               shape=[padded_sx_dim])
  ky_hr = mtf.import_tf_tensor(mesh,
                               kvec_hr[1].squeeze().astype('float32'),
                               shape=[padded_sy_dim])
  kz_hr = mtf.import_tf_tensor(mesh,
                               kvec_hr[2].squeeze().astype('float32'),
                               shape=[padded_sz_dim])
  kv_hr = [ky_hr, kz_hr, kx_hr]

  shape = [batch_dim, fx_dim, fy_dim, fz_dim]
  lr_shape = [batch_dim, x_dim, y_dim, z_dim]
  hr_shape = [batch_dim, nx_dim, ny_dim, nz_dim, sx_dim, sy_dim, sz_dim]
  part_shape = [batch_dim, fx_dim, fy_dim, fz_dim]

  # Compute initial initial conditions distributed
  initc = mtfpm.linear_field(mesh, shape, bs, nc, pk, kv)

  # Reshaping array into high resolution mesh
  field = mtf.slicewise(lambda x: tf.expand_dims(
      tf.expand_dims(tf.expand_dims(x, axis=1), axis=1), axis=1), [initc],
                        output_dtype=tf.float32,
                        output_shape=hr_shape,
                        name='my_reshape',
                        splittable_dims=lr_shape[:-1] + hr_shape[1:4] +
                        part_shape[1:3])

  for block_size_dim in hr_shape[-3:]:
    field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)

  for blocks_dim, block_size_dim in zip(hr_shape[1:4], field.shape[-3:]):
    field = mpm.halo_reduce(field, blocks_dim, block_size_dim, halo_size)

  field = mtf.reshape(field, field.shape + [mtf.Dimension('h_dim', 1)])
  high = field
  low = mesh_utils.downsample(field, downsampling_factor, antialias=True)

  low = mtf.reshape(low, low.shape[:-1])
  high = mtf.reshape(high, high.shape[:-1])

  for block_size_dim in hr_shape[-3:]:
    low = mtf.slice(low, halo_size // 2**downsampling_factor,
                    block_size_dim.size // 2**downsampling_factor,
                    block_size_dim.name)
  # Hack usisng  custom reshape because mesh is pretty dumb
  low = mtf.slicewise(lambda x: x[:, 0, 0, 0], [low],
                      output_dtype=tf.float32,
                      output_shape=lr_shape,
                      name='my_dumb_reshape',
                      splittable_dims=lr_shape[:-1] + hr_shape[:4])

  state = mtfpm.lpt_init(
        low,
        high,
        0.1,
        kv_lr,
        kv_hr,
        halo_size,
        hr_shape,
        lr_shape,
        part_shape[1:],
        downsampling_factor=downsampling_factor,
        antialias=True,
    )

  final_state = mtfpm.nbody(state,
                                stages,
                                lr_shape,
                                hr_shape,
                                kv_lr,
                                kv_hr,
                                halo_size,
                                downsampling_factor=downsampling_factor)

  # paint the field
  final_field = mtf.zeros(mesh, shape=hr_shape)
  for block_size_dim in hr_shape[-3:]:
    final_field = mtf.pad(final_field, [halo_size, halo_size],
                          block_size_dim.name)
  final_field = mesh_utils.cic_paint(final_field, final_state[0], halo_size)
  # Halo exchange
  for blocks_dim, block_size_dim in zip(hr_shape[1:4], final_field.shape[-3:]):
    final_field = mpm.halo_reduce(final_field, blocks_dim, block_size_dim,
                                  halo_size)
  # Remove borders
  for block_size_dim in hr_shape[-3:]:
    final_field = mtf.slice(final_field, halo_size, block_size_dim.size,
                            block_size_dim.name)

  final_field = mtf.slicewise(lambda x: x[:, 0, 0, 0], [final_field],
                              output_dtype=tf.float32,
                              output_shape=[batch_dim, fx_dim, fy_dim, fz_dim],
                              name='my_dumb_reshape',
                              splittable_dims=part_shape[:-1] + hr_shape[:4])

  return initc, final_field


def main(_):

  # Creating layout and mesh implementation
  mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]
  layout_rules = [("nx_lr", "row"), ("ny_lr", "col"), ("nx", "row"),
                  ("ny", "col"), ("ty", "row"), ("tz", "col"),
                  ("ty_lr", "row"), ("tz_lr", "col"), ("nx_block", "row"),
                  ("ny_block", "col")]
  mesh_impl = HvdSimdMeshImpl(mtf.convert_to_shape(mesh_shape), 
                              mtf.convert_to_layout_rules(layout_rules))

  # Create the graph and mesh
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")

  ## Load initial power spectrum
  klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
  
  # Defines the computational graph for the nbody
  initial_conditions, final_field = nbody_fn(mesh, klin, plin)

  # Lower mesh computation
  lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  
  # Retrieve fields as tf tensors
  tf_initc = lowering.export_to_tf_tensor(initial_conditions)
  tf_final = lowering.export_to_tf_tensor(final_field)
  
  with tf.Session() as sess:
    start = time.time()
    init_conds, final = sess.run([tf_initc, tf_final])
    end = time.time()
    print('\n Time for the mesh run : %f \n' % (end - start))

  # Export these fields
  np.save('simulation_output_%d.npy'%comm.Get_rank(), final)
  np.save('simulation_input_%d.npy'%comm.Get_rank(), init_conds)

  exit(0)


if __name__ == "__main__":
  tf.app.run(main=main)
