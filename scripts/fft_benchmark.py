"""
Benchmark script for studying the scaling of distributed FFTs on Mesh Tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import flowpm.mesh_ops as mpm

tf.flags.DEFINE_integer("num_iters", 10, "Number of FFT transforms.")
tf.flags.DEFINE_integer("cube_size", 512, "Size of the 3D volume.")
tf.flags.DEFINE_integer("batch_size", 64,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_string("mesh_shape", "b1:4;b2:4", "mesh shape")
tf.flags.DEFINE_string("layout", "x:b1;y:b2",
                       "layout rules")

def benchmark_model(mesh):
  """
  Initializes a 3D volume with random noise, and execute a forward FFT
  """
  batch_dim = mtf.Dimension("batch", FLAGS.cube_size)
  x_dim = mtf.Dimension("nx", FLAGS.cube_size)
  y_dim = mtf.Dimension("ny", FLAGS.cube_size)
  z_dim = mtf.Dimension("nz", FLAGS.cube_size)

  # Create field
  field = mtf.random_uniform(mesh, [batch_dim, x_dim, y_dim, z_dim])

  # Apply FFT
  fft_field = mpm.fft3d(field)

  # Inverse FFT
  rfield = mtf.cast(mpm.ifft3d(fft_field), tf.float32)

  # Compute errors
  err = mtf.reduce_max(mtf.abs(field - rfield))
  return err


def main(_):
  # Get MPI rank, we assume one process by node
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Retrieve the list of nodes from SLURM
  # Parse them with ridiculous logic
  hosts_list=os.environ['SLURM_NODELIST']
  hosts_list = ["cgpu"+ s for s in hosts_list.split("cgpu[")[1].split("]")[0].replace('-',',').split(",")]
  mesh_hosts = [hosts_list[i] + ":%d"%(8222+j) for i in range(len(hosts_list)) for j in range(1)]

  if rank ==0 :
      print(hosts_list)

  # Create a cluster from the mesh hosts.
  cluster = tf.train.ClusterSpec({"mesh": mesh_hosts})

  # Create a server for local mesh members
  server = tf.train.Server(cluster,
                           job_name="mesh",
                           task_index=rank)

  # Only he master job takes care of the graph building
  if rank >0:
      server.join()

  # Otherwise we are the main task, let's define the devices
  mesh_devices = ["/job:mesh/task:%d/device:GPU:%d"%(i,j) for i in range(cluster.num_tasks("mesh")) for j in range(8)]
  print("List of devices", mesh_devices)

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "fft_mesh")
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)

  # Build the model
  fft_err = benchmark_model(mesh)

  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  # Retrieve output of computation
  result = lowering.export_to_tf_tensor(fft_err)

  with tf.Session(server.target) as sess:
    err = sess.run(result)
    start = time.time()
    for _ in range(FLAGS.num_iters):
      err = sess.run(result)
    end = time.time()

  print("Max absolute FFT error %f, with wall time %f"%(err, (end - start) / num_iters))
  exit(0)


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
