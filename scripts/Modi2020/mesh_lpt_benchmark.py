import numpy as np
import os
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf

import time
import sys
sys.path.append('../')
sys.path.append('../flowpm/')
import flowpm.mesh_ops as mpm
import flowpm.mtfpm as mtfpm
import flowpm.mesh_utils as mesh_utils
import flowpm
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt
cosmology = Planck15

tf.flags.DEFINE_integer("gpus_per_node", 8, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 8, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")

tf.flags.DEFINE_integer("nc", 128, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 500, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")

#pyramid flags
tf.flags.DEFINE_integer("dsample", 0, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 0, "halo size")

#mesh flags
tf.flags.DEFINE_integer("nx", 4, "# blocks along x")
tf.flags.DEFINE_integer("ny", 2, "# blocks along y")
tf.flags.DEFINE_string("mesh_shape", "row:16", "mesh shape")
#tf.flags.DEFINE_string("layout", "nx:b1", "layout rules")
tf.flags.DEFINE_string("output_file", "timeline",
                       "Name of the output timeline file")

FLAGS = tf.flags.FLAGS


def lpt_prototype(mesh,
                  nc=FLAGS.nc,
                  bs=FLAGS.box_size,
                  batch_size=FLAGS.batch_size,
                  a0=FLAGS.a0,
                  a=FLAGS.af,
                  nsteps=FLAGS.nsteps):
  """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """

  klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
  plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
  ipklin = iuspline(klin, plin)
  stages = np.linspace(a0, a, nsteps, endpoint=True)

  # Define the named dimensions
  # Parameters of the small scales decomposition
  n_block_x = FLAGS.nx
  n_block_y = FLAGS.ny
  n_block_z = 1
  halo_size = FLAGS.hsize

  if halo_size >= 0.5 * min(nc // n_block_x, nc // n_block_y, nc // n_block_z):
    new_size = int(0.5 *
                   min(nc // n_block_x, nc // n_block_y, nc // n_block_z))
    print('WARNING: REDUCING HALO SIZE from %d to %d' % (halo_size, new_size))
    halo_size = new_size

  # Parameters of the large scales decomposition
  downsampling_factor = 0
  lnc = nc // 2**downsampling_factor

  #

  fx_dim = mtf.Dimension("nx", nc)
  fy_dim = mtf.Dimension("ny", nc)
  fz_dim = mtf.Dimension("nz", nc)

  tfx_dim = mtf.Dimension("tx", nc)
  tfy_dim = mtf.Dimension("ty", nc)
  tfz_dim = mtf.Dimension("tz", nc)

  tx_dim = mtf.Dimension("tx_lr", nc)
  ty_dim = mtf.Dimension("ty_lr", nc)
  tz_dim = mtf.Dimension("tz_lr", nc)

  nx_dim = mtf.Dimension('nx_block', n_block_x)
  ny_dim = mtf.Dimension('ny_block', n_block_y)
  nz_dim = mtf.Dimension('nz_block', n_block_z)

  sx_dim = mtf.Dimension('sx_block', nc // n_block_x)
  sy_dim = mtf.Dimension('sy_block', nc // n_block_y)
  sz_dim = mtf.Dimension('sz_block', nc // n_block_z)

  k_dims = [tx_dim, ty_dim, tz_dim]

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
  kvec_lr = flowpm.kernels.fftk([nc, nc, nc], symmetric=False)
  kx_lr = mtf.import_tf_tensor(mesh,
                               kvec_lr[0].squeeze().astype('float32'),
                               shape=[tx_dim])
  ky_lr = mtf.import_tf_tensor(mesh,
                               kvec_lr[1].squeeze().astype('float32'),
                               shape=[ty_dim])
  kz_lr = mtf.import_tf_tensor(mesh,
                               kvec_lr[2].squeeze().astype('float32'),
                               shape=[tz_dim])
  kv_lr = [ky_lr, kz_lr, kx_lr]

  shape = [batch_dim, fx_dim, fy_dim, fz_dim]
  lr_shape = [batch_dim, fx_dim, fy_dim, fz_dim]
  hr_shape = [batch_dim, nx_dim, ny_dim, nz_dim, sx_dim, sy_dim, sz_dim]
  part_shape = [batch_dim, fx_dim, fy_dim, fz_dim]

  # Begin simulation

  initc = mtfpm.linear_field(mesh, shape, bs, nc, pk, kv)

  #    # Reshaping array into high resolution mesh
  #    field = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),axis=1),
  #                      [initc],
  #                      output_dtype=tf.float32,
  #                      output_shape=hr_shape,
  #                      name='my_reshape',
  #                      splittable_dims=lr_shape[:-1]+hr_shape[1:4]+part_shape[1:3])
  #

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
  # Here we can run our nbody
  final_state = state  #mtfpm.nbody(state, stages, lr_shape, hr_shape, k_dims, kv_lr, kv_hr, halo_size, downsampling_factor=downsampling_factor)

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
  final_field = mtf.slicewise(lambda x: x[:, 0, 0, 0], [final_field],
                              output_dtype=tf.float32,
                              output_shape=[batch_dim, fx_dim, fy_dim, fz_dim],
                              name='my_dumb_reshape',
                              splittable_dims=part_shape[:-1] + hr_shape[:4])

  return initc, final_field

  ##


def main(_):

  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)

  print(mesh_shape)

  #layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
  #mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]
  layout_rules = [("nx_lr", "row"), ("ny_lr", "col"), ("nx", "row"),
                  ("ny", "col"), ("ty", "row"), ("tz", "col"),
                  ("ty_lr", "row"), ("tz_lr", "col"), ("nx_block", "row"),
                  ("ny_block", "col")]

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
      for j in range(FLAGS.gpus_per_task)
  ]
  print("List of devices", mesh_devices)

  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, layout_rules, mesh_devices)

  # Build the model
  # Create computational graphs and some initializations

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "nbody_mesh")

  initial_conditions, mesh_final_field = lpt_prototype(
      mesh, bs=FLAGS.box_size, nc=FLAGS.nc, batch_size=FLAGS.batch_size)
  # Lower mesh computation
  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  # Retrieve output of computation
  initc = lowering.export_to_tf_tensor(initial_conditions)
  result = lowering.export_to_tf_tensor(mesh_final_field)

  with tf.Session(server.target,
                  config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False)) as sess:
    a, c = sess.run([initc, result])

  plt.figure(figsize=(9, 3))
  plt.subplot(121)
  plt.imshow(a[0].sum(axis=2))
  plt.title('Initial Conditions')

  plt.subplot(122)
  plt.imshow(c[0].sum(axis=2))
  plt.title('Mesh TensorFlow')
  plt.colorbar()
  plt.savefig("figs/mesh_lpt_%d-b1%d-b2%d.png" %
              (FLAGS.nc, FLAGS.nx, FLAGS.ny))

  with tf.Session(server.target) as sess:

    start = time.time()
    err = sess.run(result)
    end = time.time()

    niter = 10
    start = time.time()
    for i in range(niter):
      err = sess.run(result)
    end = time.time()
    ttime = (end - start) / niter


#        profiler = tf.profiler.Profiler(sess.graph)
#
#        run_meta = tf.RunMetadata()
#        err = sess.run(result,
#                       options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
#                       run_metadata=run_meta)
#
#        profiler.add_step(0, run_meta)
#
#        opts = (tf.profiler.ProfileOptionBuilder(
#            tf.profiler.ProfileOptionBuilder.time_and_memory())
#            .with_step(0)
#            .with_timeline_output(FLAGS.output_file).build())
#        profiler.profile_graph(options=opts)
#

  print('Time for ', mesh_shape, ' is : ', ttime)
  exit(-1)

if __name__ == "__main__":
  tf.app.run(main=main)
