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
import flowpm.mesh_raytracing as mesh_raytracing
import flowpm
from astropy.cosmology import Planck15
##

cosmology = Planck15
tf.random.set_random_seed(200*comm.Get_rank())

tf.flags.DEFINE_integer("nc", 256, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 256, "Box Size [Mpc/h]")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")

# Ray tracing flags
tf.flags.DEFINE_integer("lensplane_nc", 512, "Size of the lens planes")
tf.flags.DEFINE_float("field_size", 5., "Size of the lensing field in degrees")
tf.flags.DEFINE_integer("field_npix", 512, "Number of pixels in the lensing field")
tf.flags.DEFINE_integer("n_lens", 22, "Number of lensplanes in the lightcone")

#pyramid flags
tf.flags.DEFINE_integer("dsample", 2, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 32, "halo size")

#mesh flags
tf.flags.DEFINE_integer("nx", 2, "# blocks along x")
tf.flags.DEFINE_integer("ny", 2, "# blocks along y")

FLAGS = tf.flags.FLAGS

def nbody_fn(mesh,
             klin, 
             plin,
             stages,
             n_stages,
             nc=FLAGS.nc,
             bs=FLAGS.box_size,
             batch_size=FLAGS.batch_size,
             a0=FLAGS.a0,
             a=FLAGS.af,
             lensplane_nc=FLAGS.lensplane_nc,
             dtype=tf.float32):
  """ Pyramid N-body function
  """

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

  # Dimensions for the lens planes
  lpx_dim = mtf.Dimension('nx_lp', lensplane_nc)
  lpy_dim = mtf.Dimension('ny_lp', lensplane_nc)
  
  slpx_dim = mtf.Dimension('sx_block_lp', lensplane_nc // n_block_x)
  slpy_dim = mtf.Dimension('sy_block_lp', lensplane_nc // n_block_y)


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
  lp_shape = [batch_dim, nx_dim, ny_dim, slpx_dim, slpy_dim]

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
        a0,
        kv_lr,
        kv_hr,
        halo_size,
        hr_shape,
        lr_shape,
        part_shape[1:],
        downsampling_factor=downsampling_factor,
        antialias=True,
    )

  states = mtfpm.nbody(state,
                       stages,
                       n_stages,
                       lr_shape,
                       hr_shape,
                       kv_lr,
                       kv_hr,
                       halo_size,
                       downsampling_factor=downsampling_factor,
                       return_intermediate_states=True)

  # Extract lensplanes
  lensplanes = []
  for i in range(n_stages-1):
    plane = mesh_raytracing.density_plane(
        states[::-1][i][1],
        FLAGS.nc,
        plane_resolution=lensplane_nc,
        halo_size=halo_size, 
        lp_shape=lp_shape)
    # Remove split axis
    plane = mtf.slicewise(lambda x: x[:, 0, 0], [plane],
                          output_dtype=tf.float32,
                          output_shape=[batch_dim, lpx_dim, lpy_dim],
                          name='my_dumb_reshape',
                          splittable_dims=lp_shape+[lpx_dim, lpy_dim])

    # Anonymize and export
    plane = mtf.anonymize(plane)
    lensplanes.append((states[::-1][i][0], plane))

  return lensplanes


def main(_):

  # Creating layout and mesh implementation
  mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]
  layout_rules = [("nx_lr", "row"), ("ny_lr", "col"), ("nx", "row"),
                  ("ny", "col"), ("ty", "row"), ("tz", "col"),
                  ("ty_lr", "row"), ("tz_lr", "col"), ("nx_block", "row"),
                  ("ny_block", "col"),
                  ("nx_lp", "row"), ("ny_lp", "col")]
  mesh_impl = HvdSimdMeshImpl(mtf.convert_to_shape(mesh_shape), 
                              mtf.convert_to_layout_rules(layout_rules))

  # Create the graph and mesh
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")

  ## Load initial power spectrum
  klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
  
  # Instantiates a cosmology with desired parameters
  cosmology = flowpm.cosmology.Planck15()

  # Schedule the center of the lensplanes we want for ray tracing
  r = tf.linspace(0., FLAGS.box_size * FLAGS.n_lens, FLAGS.n_lens + 1)
  r_center = 0.5 * (r[1:] + r[:-1])

  # Retrieve the scale factor corresponding to these distances
  a = flowpm.tfbackground.a_of_chi(cosmology, r)
  a_center = flowpm.tfbackground.a_of_chi(cosmology, r_center)

  # We run 5 steps from initial scale factor to start of raytracing
  init_stages = tf.linspace(FLAGS.a0, a[-1], 5)
  # Then one step per lens plane
  stages = tf.concat([init_stages, a_center[::-1]], axis=0)
  
  with tf.Session() as sess:
    stages, r_center, a_center = sess.run([stages, r_center, a_center])

  n_stages = 5 + FLAGS.n_lens
  # Defines the computational graph for the nbody
  mesh_lensplanes = nbody_fn(mesh, klin, plin, stages, n_stages)

  # Lower mesh computation
  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  lensplanes = []
  for i in range(len(mesh_lensplanes)):
    plane = lowering.export_to_tf_tensor(mesh_lensplanes[i][1])
    print("expected vs found", a_center[i], mesh_lensplanes[i][0])
    # Apply random shuffling
    plane = tf.expand_dims(plane, axis=-1)
    plane = tf.image.random_flip_left_right(plane)
    plane = tf.image.random_flip_up_down(plane)
    shift_x = np.random.randint(0, FLAGS.lensplane_nc -1)
    shift_y = np.random.randint(0, FLAGS.lensplane_nc -1)
    plane = tf.roll(plane, shift=[shift_x, shift_y], axis=[1,2])
    lensplanes.append((r_center[i], a_center[i], plane[..., 0]))

  # And now, interpolate and ray trace
  xgrid, ygrid = np.meshgrid(
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False),  # range of X coordinates
      np.linspace(0, FLAGS.field_size, FLAGS.field_npix,
                  endpoint=False))  # range of Y coordinates

  coords = np.stack([xgrid, ygrid], axis=0) 
  c = coords.reshape([2, -1]).T / 180.*np.pi # to radians
  # Create array of source redshifts
  z_source = tf.linspace(0.5, 1, 4)
  m = flowpm.raytracing.convergenceBorn(cosmology,
                                        lensplanes,
                                        dx=FLAGS.box_size / FLAGS.lensplane_nc,
                                        dz=FLAGS.box_size,
                                        coords=c,
                                        z_source=z_source)
  m = tf.reshape(m, [1, FLAGS.field_npix, FLAGS.field_npix, -1])

  
  with tf.Session() as sess:
    start = time.time()
    maps = sess.run(m)
    end = time.time()
    print('\n Time for the mesh run : %f \n' % (end - start))

  # Export these fields
  np.save('convergence_maps_%d.npy'%comm.Get_rank(), maps)

  exit(0)


if __name__ == "__main__":
  tf.app.run(main=main)
