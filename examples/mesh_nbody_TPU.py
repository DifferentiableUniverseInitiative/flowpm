"""
Script running an N-body PM simulation on TPU
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.lib.io import file_io
import mesh_tensorflow as mtf
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib

import flowpm
import flowpm.mesh_utils as mpu
import flowpm.mtfpm as fpm

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

tf.flags.DEFINE_integer("cube_size", 512, "Size of the 3D volume.")
tf.flags.DEFINE_float("box_size", 1000., "Physical size of the 3D volume.")
tf.flags.DEFINE_float("a0", 0.1, "Scale factor of linear field.")
tf.flags.DEFINE_integer("pm_steps", 10, "Number of PM steps.")

tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")

tf.flags.DEFINE_string("mesh_shape", "b1:32", "mesh shape")
tf.flags.DEFINE_string("layout", "nx:b1", "layout rules")

FLAGS = tf.flags.FLAGS

def nbody_model(mesh):
  """
  Initializes a 3D volume with random noise, and execute a forward FFT
  """

  # Initialize the integration steps
  stages = np.linspace(FLAGS.a0, 1.0, FLAGS.pm_steps, endpoint=True)

  # Define required dimensions for the problem
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  x_dim = mtf.Dimension("nx", FLAGS.cube_size)
  y_dim = mtf.Dimension("ny", FLAGS.cube_size)
  z_dim = mtf.Dimension("nz", FLAGS.cube_size)

  # Load the power spectrum for initial conditions
  klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
  pk_dim = mtf.Dimension("npk", len(plin))
  pk = mtf.import_tf_tensor(mesh, plin.astype('float32'), shape=[pk_dim])

  # Initialize the fftfreq vector for the Fourier Transforms
  kvec = flowpm.kernels.fftk([FLAGS.cube_size, FLAGS.cube_size, FLAGS.cube_size],
                             symmetric=False)
  kx = mtf.import_tf_tensor(mesh, kvec[0].squeeze().astype('float32'), shape=[x_dim])
  ky = mtf.import_tf_tensor(mesh, kvec[1].squeeze().astype('float32'), shape=[y_dim])
  kz = mtf.import_tf_tensor(mesh, kvec[2].squeeze().astype('float32'), shape=[z_dim])
  kv = [kx, ky, kz]

  #### N-body simulation starts here
  # Create initial conditions
  initial_conditions = fpm.linear_field(mesh, [batch_dim, x_dim, y_dim, z_dim],
                                        FLAGS.box_size, pk, kv)

  # LPT evolution to a0
  state = fpm.lpt_init(initial_conditions, FLAGS.a0, kv, [x_dim], [32])

  # Nbody all the way to a=1
  final_state = fpm.nbody(state,  stages, [batch_dim, x_dim, y_dim, z_dim], kv,  [x_dim], [32])

  # Let's paint the result back on the grid for visualization
  final_field = mpu.cic_paint(mtf.zeros_like(initial_conditions), final_state[0],  [x_dim], [32])
  #### Done :-)


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

    # Until we implement distributed outputs, we only return one example
    field_slice, _ = mtf.split(field, batch_dim, [1, FLAGS.batch_size-1])
    field_slice = mtf.reshape(field_slice, [mtf.Dimension("bs", 1), x_dim_nosplit, y_dim, z_dim])

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
