"""
Benchmark script for studying the scaling of distributed FFTs on Mesh Tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
import flowpm.mesh_ops as mpm
from tensorflow.python.client import timeline

tf.flags.DEFINE_integer("gpus_per_node", 2, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 2, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")

tf.flags.DEFINE_integer("cube_size", 512, "Size of the 3D volume.")
tf.flags.DEFINE_integer(
    "batch_size", 64, "Mini-batch size for the training. Note that this "
    "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_string("mesh_shape", "b1:16", "mesh shape")
tf.flags.DEFINE_string("layout", "nx:b1", "layout rules")

tf.flags.DEFINE_integer(
    "n_ffts", 10, "Number of back and forth FFTs in single"
    "session run.")

FLAGS = tf.flags.FLAGS


def benchmark_model(mesh):
  """
  Initializes a 3D volume with random noise, and execute a forward FFT
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  x_dim = mtf.Dimension("nx", FLAGS.cube_size)
  y_dim = mtf.Dimension("ny", FLAGS.cube_size)
  z_dim = mtf.Dimension("nz", FLAGS.cube_size)

  tx_dim = mtf.Dimension("tnx", FLAGS.cube_size)
  ty_dim = mtf.Dimension("tny", FLAGS.cube_size)
  tz_dim = mtf.Dimension("tnz", FLAGS.cube_size)

  # Create field
  field = mtf.random_normal(mesh, [batch_dim, x_dim, y_dim, z_dim])

  input_field = field
  field = mtf.cast(field, tf.complex64)
  err = 0
  # Performs several back and forth FFTs in the same session
  for i in range(FLAGS.n_ffts):
    # Apply FFT
    fft_field = mpm.fft3d(field, [tx_dim, ty_dim, tz_dim])
    # Inverse FFT
    field = mpm.ifft3d(fft_field * 1, [x_dim, y_dim, z_dim])
    err += mtf.reduce_max(mtf.abs(mtf.cast(field, tf.float32) - input_field))

  field = mtf.cast(field, tf.float32)
  # Compute errors
  err += mtf.reduce_max(mtf.abs(field - input_field))
  return err


def main(_):

  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)

  # Resolve the cluster from SLURM environment
  cluster = tf.distribute.cluster_resolver.SlurmClusterResolver(
      {"mesh": mesh_shape.size // FLAGS.gpus_per_task},
      port_base=8822,
      gpus_per_node=FLAGS.gpus_per_node,
      gpus_per_task=FLAGS.gpus_per_task,
      tasks_per_node=FLAGS.tasks_per_node)

  cluster_spec = cluster.cluster_spec()
  # Create a server for all mesh members
  server = tf.distribute.Server(cluster_spec, "mesh", cluster.task_id)

  # Only he master job takes care of the graph building,
  # everyone else can just chill for now
  if cluster.task_id > 0:
    server.join()

  # Otherwise we are the main task, let's define the devices
  mesh_devices = [
      "/job:mesh/task:%d/device:GPU:%d" % (i, j)
      for i in range(cluster_spec.num_tasks("mesh"))
      for j in range(FLAGS.gpus_per_node)
  ]
  print("List of devices", mesh_devices)

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "fft_mesh")

  # Build the model
  fft_err = benchmark_model(mesh)

  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  # Retrieve output of computation
  result = lowering.export_to_tf_tensor(fft_err)

  with tf.Session(server.target) as sess:
    start = time.time()
    err = sess.run(result)
    end = time.time()

    time.sleep(1)
    start = time.time()
    err = sess.run(result)
    end = time.time()

  print("Max absolute FFT error %f, with wall time %f" % (err, (end - start)))
  time.sleep(1)
  exit(0)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
