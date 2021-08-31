"""
Benchmark script for studying the scaling of distributed FFTs on Mesh Tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf

import flowpm
import flowpm.mesh_ops as mpm
import flowpm.mtfpm as mtfpm
import flowpm.mesh_utils as mesh_utils

from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu",
    default="flowpm",
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone",
    default="europe-west4-a",
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project",
    default="flowpm",
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")

tf.flags.DEFINE_integer("cube_size", 64, "Size of the 3D volume.")
tf.flags.DEFINE_float("box_size", 200., "Physical size of the 3D volume.")
tf.flags.DEFINE_float("a0", 0.1, "Scale factor of linear field.")
tf.flags.DEFINE_integer("pm_steps", 10, "Number of PM steps.")

tf.flags.DEFINE_integer(
    "batch_size", 64, "Mini-batch size for the training. Note that this "
    "is the global batch size and not the per-shard batch.")

tf.flags.DEFINE_string("mesh_shape", "b1:8,b2:4", "mesh shape")
tf.flags.DEFINE_string(
    "layout",
    "nx:b1,ny:b2,nx_lr:b1,ny_lr:b2,ty:b1,tz:b2,ty_lr:b1,tz_lr:b2,nx_block:b1,ny_block:b2",
    "layout rules")

FLAGS = tf.flags.FLAGS


def benchmark_model(mesh):
  """
  Initializes a 3D volume with random noise, and execute a forward FFT
  """
  # Setup parameters
  bs = FLAGS.box_size
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
  initial_conditions = flowpm.linear_field(
      nc,  # size of the cube
      bs,  # Physical size of the cube
      ipklin,  # Initial power spectrum
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
  halo_size = 4

  # Parameters of the large scales decomposition
  downsampling_factor = 2
  lnc = nc // 2**downsampling_factor

  fx_dim = mtf.Dimension("nx", nc)
  fy_dim = mtf.Dimension("ny", nc)
  fz_dim = mtf.Dimension("nz", nc)

  tfx_dim = mtf.Dimension("tx", nc)
  tfy_dim = mtf.Dimension("ty", nc)
  tfz_dim = mtf.Dimension("tz", nc)

  # Dimensions of the low resolution grid
  tx_dim = mtf.Dimension("tx_lr", nc)
  ty_dim = mtf.Dimension("ty_lr", nc)
  tz_dim = mtf.Dimension("tz_lr", nc)

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
  kx = mtf.import_tf_tensor(
      mesh, kvec[0].squeeze().astype('float32'), shape=[tfx_dim])
  ky = mtf.import_tf_tensor(
      mesh, kvec[1].squeeze().astype('float32'), shape=[tfy_dim])
  kz = mtf.import_tf_tensor(
      mesh, kvec[2].squeeze().astype('float32'), shape=[tfz_dim])
  kv = [ky, kz, kx]

  kvec_lr = flowpm.kernels.fftk([nc, nc, nc], symmetric=False)
  kx_lr = mtf.import_tf_tensor(
      mesh, kvec_lr[0].squeeze().astype('float32'), shape=[tx_dim])
  ky_lr = mtf.import_tf_tensor(
      mesh, kvec_lr[1].squeeze().astype('float32'), shape=[ty_dim])
  kz_lr = mtf.import_tf_tensor(
      mesh, kvec_lr[2].squeeze().astype('float32'), shape=[tz_dim])
  kv_lr = [ky_lr, kz_lr, kx_lr]

  # kvec for high resolution blocks
  shape = [batch_dim, fx_dim, fy_dim, fz_dim]
  lr_shape = [batch_dim, fx_dim, fy_dim, fz_dim]
  hr_shape = [batch_dim, nx_dim, ny_dim, nz_dim, sx_dim, sy_dim, sz_dim]
  part_shape = [batch_dim, fx_dim, fy_dim, fz_dim]

  initc = mtfpm.linear_field(mesh, shape, bs, nc, pk, kv)
  state = mtfpm.lpt_init_single(
      initc,
      a0,
      kv_lr,
      halo_size,
      lr_shape,
      hr_shape,
      part_shape[1:],
      antialias=True,
  )
  #state = mtfpm.lpt_init(low, high, 0.1, kv_lr, kv_hr, halo_size, hr_shape, lr_shape,
  #                       part_shape[1:], downsampling_factor=downsampling_factor, antialias=True,)

  # Here we can run our nbody
  final_state = state  #mtfpm.nbody(state, stages, lr_shape, hr_shape, kv_lr, kv_hr, halo_size, downsampling_factor=downsampling_factor)

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

  #final_field = mtf.reshape(final_field,  [batch_dim, fx_dim, fy_dim, fz_dim])
  # Hack usisng  custom reshape because mesh is pretty dumb
  final_field = mtf.slicewise(
      lambda x: x[:, 0, 0, 0], [final_field],
      output_dtype=tf.float32,
      output_shape=[batch_dim, fx_dim, fy_dim, fz_dim],
      name='my_dumb_reshape',
      splittable_dims=part_shape[:-1] + hr_shape[:4])

  return mtf.reduce_sum(final_field)


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
  mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(mesh_shape, layout_rules,
                                              mesh_devices,
                                              ctx.device_assignment)

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "fft_mesh")

  with mtf.utils.outside_all_rewrites():
    fsum = benchmark_model(mesh)
  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  tf_err = tf.to_float(lowering.export_to_tf_tensor(fsum))

  with mtf.utils.outside_all_rewrites():
    return tpu_estimator.TPUEstimatorSpec(mode, loss=tf_err)


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)

  # Resolve the TPU environment
  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

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
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)

  def dummy_input_fn(params):
    """Dummy input function """
    return tf.zeros(
        shape=[params['batch_size']], dtype=tf.float32), tf.zeros(
            shape=[params['batch_size']], dtype=tf.float32)

  # Run evaluate loop for ever, we will be connecting to this process using a profiler
  model.evaluate(input_fn=dummy_input_fn, steps=10000)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
