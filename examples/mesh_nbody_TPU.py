"""
Script running an N-body PM simulation on TPU
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.lib.io import file_io
import mesh_tensorflow as mtf
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

import flowpm
import flowpm.mesh_ops as mpm
import flowpm.mtfpm as mtfpm
import flowpm.mesh_utils as mesh_utils

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default="flowpm",
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default="europe-west4-a",
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default="flowpm",
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_string("output_dir", None, "Output directory for simulations")

tf.flags.DEFINE_integer("cube_size", 256, "Size of the 3D volume.")
tf.flags.DEFINE_float("box_size", 1000., "Physical size of the 3D volume.")
tf.flags.DEFINE_float("a0", 0.1, "Scale factor of linear field.")
tf.flags.DEFINE_integer("pm_steps", 10, "Number of PM steps.")

tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")

tf.flags.DEFINE_string("mesh_shape", "b1:8,b2:4", "mesh shape")
tf.flags.DEFINE_string("layout", "nx:b1,ny:b2,nx_lr:b1,ny_lr:b2,tny_lr:b1,tnz_lr:b2,nx_block:b1,ny_block:b2", "layout rules")

FLAGS = tf.flags.FLAGS

def nbody_model(mesh):
  """
  Initializes a 3D volume with random noise, and execute a forward FFT
  """
  # Setup parameters
  nc = FLAGS.cube_size
  batch_size = FLAGS.batch_size
  a0 = FLAGS.a0
  a = 1.0
  nsteps = FLAGS.pm_steps

  # Compute a few things first, using simple tensorflow
  klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)
  stages = np.linspace(a0, a, nsteps, endpoint=True)

  # Initialize the integration steps
  stages = np.linspace(FLAGS.a0, 1.0, FLAGS.pm_steps, endpoint=True)

  # Generate a batch of 3D initial conditions
  initial_conditions = flowpm.linear_field(nc,          # size of the cube
                                           bs,          # Physical size of the cube
                                           ipklin,      # Initial power spectrum
                                           batch_size=batch_size)

  # Compute necessary Fourier kernels
  kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False)
  from flowpm.kernels import laplace_kernel, gradient_kernel
  lap = tf.cast(laplace_kernel(kvec), tf.complex64)
  grad_x = gradient_kernel(kvec, 0)
  grad_y = gradient_kernel(kvec, 1)
  grad_z = gradient_kernel(kvec, 2)

  # Define the named dimensions
  # Parameters of the small scales decomposition
  n_block_x = 8
  n_block_y = 4
  n_block_z = 1
  halo_size = 16

  # Parameters of the large scales decomposition
  downsampling_factor = 2
  lnc = nc // 2**downsampling_factor

  fx_dim = mtf.Dimension("nx", nc)
  fy_dim = mtf.Dimension("ny", nc)
  fz_dim = mtf.Dimension("nz", nc)

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

  sx_dim = mtf.Dimension('sx_block', nc//n_block_x)
  sy_dim = mtf.Dimension('sy_block', nc//n_block_y)
  sz_dim = mtf.Dimension('sz_block', nc//n_block_z)

  k_dims = [tx_dim, ty_dim, tz_dim]

  batch_dim = mtf.Dimension("batch", batch_size)
  pk_dim = mtf.Dimension("npk", len(plin))
  pk = mtf.import_tf_tensor(mesh, plin.astype('float32'), shape=[pk_dim])

  # kvec for low resolution grid
  kvec_lr = flowpm.kernels.fftk([lnc, lnc, lnc], symmetric=False)

  kx_lr = mtf.import_tf_tensor(mesh, kvec_lr[0].squeeze().astype('float32')/ 2**downsampling_factor, shape=[tx_dim])
  ky_lr = mtf.import_tf_tensor(mesh, kvec_lr[1].squeeze().astype('float32')/ 2**downsampling_factor, shape=[ty_dim])
  kz_lr = mtf.import_tf_tensor(mesh, kvec_lr[2].squeeze().astype('float32')/ 2**downsampling_factor, shape=[tz_dim])
  kv_lr = [ky_lr, kz_lr, kx_lr]

  # kvec for high resolution blocks
  padded_sx_dim = mtf.Dimension('padded_sx_block', nc//n_block_x+2*halo_size)
  padded_sy_dim = mtf.Dimension('padded_sy_block', nc//n_block_y+2*halo_size)
  padded_sz_dim = mtf.Dimension('padded_sz_block', nc//n_block_z+2*halo_size)
  kvec_hr = flowpm.kernels.fftk([nc//n_block_x+2*halo_size, nc//n_block_y+2*halo_size, nc//n_block_z+2*halo_size], symmetric=False)

  kx_hr = mtf.import_tf_tensor(mesh, kvec_hr[0].squeeze().astype('float32'), shape=[padded_sx_dim])
  ky_hr = mtf.import_tf_tensor(mesh, kvec_hr[1].squeeze().astype('float32'), shape=[padded_sy_dim])
  kz_hr = mtf.import_tf_tensor(mesh, kvec_hr[2].squeeze().astype('float32'), shape=[padded_sz_dim])
  kv_hr = [kx_hr, ky_hr, kz_hr]

  lr_shape = [batch_dim, x_dim, y_dim, z_dim]

  hr_shape = [batch_dim, nx_dim, ny_dim, nz_dim, sx_dim, sy_dim, sz_dim]

  part_shape = [batch_dim, fx_dim, fy_dim, fz_dim]

  initc = tf.reshape(initial_conditions, [1, n_block_x, nc//n_block_x,
                                             n_block_y, nc//n_block_y,
                                          1, nc])

  initc = tf.transpose(initc, [0, 1, 3, 5, 2, 4, 6])

  field = mtf.import_tf_tensor(mesh, initc, shape=hr_shape)

  for block_size_dim in hr_shape[-3:]:
      field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)

  for blocks_dim, block_size_dim in zip(hr_shape[1:4], field.shape[-3:]):
      field = mpm.halo_reduce(field, blocks_dim, block_size_dim, halo_size)

  field = mtf.reshape(field, field.shape+[mtf.Dimension('h_dim', 1)])
  high = field
  low = mesh_utils.downsample(field, downsampling_factor, antialias=True)

  low = mtf.reshape(low, low.shape[:-1])
  high = mtf.reshape(high, high.shape[:-1])

  for block_size_dim in hr_shape[-3:]:
      low = mtf.slice(low, halo_size//2**downsampling_factor, block_size_dim.size//2**downsampling_factor, block_size_dim.name)
  # Hack usisng  custom reshape because mesh is pretty dumb
  low = mtf.slicewise(lambda x: x[:,0,0,0],
                      [low],
                      output_dtype=tf.float32,
                      output_shape=lr_shape,
                      name='my_dumb_reshape',
                      splittable_dims=lr_shape[:-1]+hr_shape[:4])

  state = mtfpm.lpt_init(low, high, 0.1, kv_lr, kv_hr, halo_size, hr_shape, lr_shape, k_dims,
                         part_shape[1:], downsampling_factor=downsampling_factor, antialias=True,)

  # Here we can run our nbody
  final_state = mtfpm.nbody(state, stages, lr_shape, hr_shape, k_dims, kv_lr, kv_hr, halo_size, downsampling_factor=downsampling_factor)

  # paint the field
  final_field = mtf.zeros(mesh, shape=hr_shape)
  for block_size_dim in hr_shape[-3:]:
      final_field = mtf.pad(final_field, [halo_size, halo_size], block_size_dim.name)
  final_field = mesh_utils.cic_paint(final_field, final_state[0], halo_size)
  # Halo exchange
  for blocks_dim, block_size_dim in zip(hr_shape[1:4], final_field.shape[-3:]):
      final_field = mpm.halo_reduce(final_field, blocks_dim, block_size_dim, halo_size)
  # Remove borders
  for block_size_dim in hr_shape[-3:]:
      final_field = mtf.slice(final_field, halo_size, block_size_dim.size, block_size_dim.name)

  #final_field = mtf.reshape(final_field,  [batch_dim, fx_dim, fy_dim, fz_dim])
   # Hack usisng  custom reshape because mesh is pretty dumb
  final_field = mtf.slicewise(lambda x: x[:,0,0,0],
                      [final_field],
                      output_dtype=tf.float32,
                      output_shape=[batch_dim, fx_dim, fy_dim, fz_dim],
                      name='my_dumb_reshape',
                      splittable_dims=part_shape[:-1]+hr_shape[:4])

  return final_field

def model_fn(features, labels, mode, params):
  """A model is called by TpuEstimator."""
  del labels
  del features

  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)

  ctx = params['context']
  num_hosts = ctx.num_hosts
  host_placement_fn = ctx.tpu_host_placement_function
  device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
  tf.logging.info('device_list = %s' % device_list,)

  mesh_devices = [''] * mesh_shape.size
  mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
      mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "fft_mesh")

  with mtf.utils.outside_all_rewrites():
    field = nbody_model(mesh)
    batch_dim, x_dim, y_dim, z_dim = field.shape
    x_dim_nosplit = mtf.Dimension("nx_nosplit", FLAGS.cube_size)
    y_dim_nosplit = mtf.Dimension("ny_nosplit", FLAGS.cube_size)

    # Until we implement distributed outputs, we only return one example
    field_slice, _ = mtf.split(field, batch_dim, [1, FLAGS.batch_size-1])
    field_slice = mtf.reshape(field_slice, [mtf.Dimension("bs", 1), x_dim_nosplit, y_dim_nosplit, z_dim])

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  tf_field = tf.to_float(lowering.export_to_tf_tensor(field_slice))

  with mtf.utils.outside_all_rewrites():
    return tpu_estimator.TPUEstimatorSpec(mode, predictions={'field': tf_field})

def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)

  # Resolve the TPU environment
  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=None,  # Disable the default saver
      save_checkpoints_secs=None,  # Disable the default saver
      log_step_count_steps=100,
      save_summary_steps=100,
      tpu_config=tpu_config.TPUConfig(
          num_shards=mesh_shape.size,
          iterations_per_loop=100,
          num_cores_per_replica=1,
          per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

  model = tpu_estimator.TPUEstimator(
      use_tpu=True,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=1,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)

  def dummy_input_fn(params):
    dset = tf.data.Dataset.from_tensor_slices(tf.zeros(shape=[params['batch_size'],1],
                                                       dtype=tf.float32))
    return dset

  # Run evaluate loop for ever, we will be connecting to this process using a profiler
  for i, f in enumerate(model.predict(input_fn=dummy_input_fn)):
    print(i)
    np.save(file_io.FileIO(FLAGS.output_dir+'/field_%d.npy'%i, 'w'), f['field'])

if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
