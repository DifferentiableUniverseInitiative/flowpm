import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import mesh_tensorflow as mtf

import flowpm
import flowpm.mesh_ops as mpm
import flowpm.mtfpm as mtfpm
import flowpm.mesh_utils as mesh_utils
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint

##


cosmology=Planck15
np.random.seed(100)
tf.random.set_random_seed(200)


tf.flags.DEFINE_integer("gpus_per_node", 8, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 8, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")

tf.flags.DEFINE_integer("nc", 128, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 500, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")
tf.flags.DEFINE_bool("nbody", False, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "", "suffix for the folder name")

#pyramid flags
tf.flags.DEFINE_integer("hsize", 32, "halo size")

#mesh flags
tf.flags.DEFINE_integer("nx", 4, "# blocks along x")
tf.flags.DEFINE_integer("ny", 2, "# blocks along y")
tf.flags.DEFINE_string("mesh_shape", "row:16", "mesh shape")
tf.flags.DEFINE_string("output_file", "timeline", "Name of the output timeline file")

FLAGS = tf.flags.FLAGS

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
stages = np.linspace(a0, a, nsteps, endpoint=True)


def nbody_prototype(mesh, infield=False, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
                        a0=FLAGS.a0, a=FLAGS.af, nsteps=FLAGS.nsteps, dtype=tf.float32):
    """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """
    # Compute a few things first, using simple tensorflow
    stages = np.linspace(a0, a, nsteps, endpoint=True)

    # Define the named dimensions
    # Parameters of the small scales decomposition
    n_block_x = FLAGS.nx
    n_block_y = FLAGS.ny
    n_block_z = 1
    halo_size = FLAGS.hsize


    # Parameters of the large scales decomposition

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

    sx_dim = mtf.Dimension('sx_block', nc//n_block_x)
    sy_dim = mtf.Dimension('sy_block', nc//n_block_y)
    sz_dim = mtf.Dimension('sz_block', nc//n_block_z)

    k_dims = [tx_dim, ty_dim, tz_dim]

    batch_dim = mtf.Dimension("batch", batch_size)

    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)
    pk_dim = mtf.Dimension("npk", len(plin))
    pk = mtf.import_tf_tensor(mesh, plin, shape=[pk_dim])

    
    # Compute necessary Fourier kernels
    kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False)
    kx = mtf.import_tf_tensor(mesh, kvec[0].squeeze().astype('float32'), shape=[tfx_dim])
    ky = mtf.import_tf_tensor(mesh, kvec[1].squeeze().astype('float32'), shape=[tfy_dim])
    kz = mtf.import_tf_tensor(mesh, kvec[2].squeeze().astype('float32'), shape=[tfz_dim])
    kv = [ky, kz, kx]

    # kvec for low resolution grid
    kvec_lr = flowpm.kernels.fftk([nc, nc, nc], symmetric=False)
    kx_lr = mtf.import_tf_tensor(mesh, kvec_lr[0].squeeze().astype('float32'), shape=[tx_dim])
    ky_lr = mtf.import_tf_tensor(mesh, kvec_lr[1].squeeze().astype('float32'), shape=[ty_dim])
    kz_lr = mtf.import_tf_tensor(mesh, kvec_lr[2].squeeze().astype('float32'), shape=[tz_dim])
    kv_lr = [ky_lr, kz_lr, kx_lr]



    shape = [batch_dim, fx_dim, fy_dim, fz_dim]
    lr_shape = [batch_dim, fx_dim, fy_dim, fz_dim]
    hr_shape = [batch_dim, nx_dim, ny_dim, nz_dim, sx_dim, sy_dim, sz_dim]
    part_shape = [batch_dim, fx_dim, fy_dim, fz_dim]



    # Begin simulation
    
    ## Compute initial initial conditions distributed
    input_field = tf.placeholder(dtype, [batch_size, nc, nc, nc])
    if infield:
        initc = mtf.import_tf_tensor(mesh, input_field, shape=part_shape)
    else:
        initc = mtfpm.linear_field(mesh, shape, bs, nc, pk, kv)

        

    # Here we can run our nbody
    if FLAGS.nbody:
        state = mtfpm.lpt_init_single(initc, a0, kv_lr, halo_size, lr_shape, hr_shape, part_shape[1:], antialias=True,)
        # Here we can run our nbody
        final_state = mtfpm.nbody_single(state, stages, lr_shape, hr_shape, kv_lr, halo_size)
    else:
        final_state = mtfpm.lpt_init_single(initc, stages[-1], kv_lr, halo_size, lr_shape, hr_shape, part_shape[1:], antialias=True,)

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

    final_field = mtf.slicewise(lambda x: x[:,0,0,0],
                        [final_field],
                        output_dtype=dtype,
                        output_shape=[batch_dim, fx_dim, fy_dim, fz_dim],
                        name='my_dumb_reshape',
                        splittable_dims=part_shape[:-1]+hr_shape[:4])

    return initc, final_field, input_field


##############################################

def main(_):

    infield = True
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)

    startw = time.time()
    
    print(mesh_shape)

    #layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
    #mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]
    layout_rules = [("nx_lr", "row"), ("ny_lr", "col"),
                    ("nx", "row"), ("ny", "col"),
                    ("ty", "row"), ("tz", "col"),
                    ("ty_lr", "row"), ("tz_lr", "col"),
                    ("nx_block","row"), ("ny_block","col")]

    # Resolve the cluster from SLURM environment
    cluster = tf.distribute.cluster_resolver.SlurmClusterResolver({"mesh": mesh_shape.size//FLAGS.gpus_per_task},
                                                                port_base=8822,
                                                                gpus_per_node=FLAGS.gpus_per_node,
                                                                gpus_per_task=FLAGS.gpus_per_task,
                                                                tasks_per_node=FLAGS.tasks_per_node)
    cluster_spec = cluster.cluster_spec()
    print(cluster_spec)
    # Create a server for all mesh members
    server = tf.distribute.Server(cluster_spec, "mesh", cluster.task_id)
    print(server)

    if cluster.task_id >0:
      server.join()

    # Otherwise we are the main task, let's define the devices
    devices = ["/job:mesh/task:%d/device:GPU:%d"%(i,j) for i in range(cluster_spec.num_tasks("mesh")) for j in range(FLAGS.gpus_per_task)]
    print("List of devices", devices)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, devices)
    
    ##Begin here
    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)

    #If initc, run normal flowpm to generate data
    tf.reset_default_graph()
    if infield:
        tfic = linear_field(FLAGS.nc, FLAGS.box_size, ipklin, batch_size=1, seed=100)
        if FLAGS.nbody:
            state = lpt_init(tfic, a0=0.1, order=1)
            final_state = nbody(state,  stages, FLAGS.nc)

        else:    
            final_state = lpt_init(tfic, a0=stages[-1], order=1)
        tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])

        start = time.time()
        with tf.Session(server.target) as sess:
            ic, fin  = sess.run([tfic, tfinal_field])
        print ("\nTime taken for the vanilla flowpm thingy :\n ", time.time()-start)

    else: ic = None

    
    
    tf.reset_default_graph()
    print('ic constructed')

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    
    initial_conditions, final_field, input_field = nbody_prototype(mesh, infield, nc=FLAGS.nc,
                                                                       batch_size=FLAGS.batch_size)

    # Lower mesh computation

    start = time.time()
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    restore_hook = mtf.MtfRestoreHook(lowering)
    end = time.time()
    print('\n Time for lowering : %f \n'%(end - start))

    
    tf_initc = lowering.export_to_tf_tensor(initial_conditions)
    tf_final = lowering.export_to_tf_tensor(final_field)
    nc = FLAGS.nc

    with tf.Session(server.target) as sess:
                    
        start = time.time()
        if infield:
            ic_check, fin_check = sess.run([tf_initc, tf_final], feed_dict={input_field:ic})
        else:
            ic_check, fin_check = sess.run([tf_initc, tf_final])
            ic, fin = ic_check, fin_check    
        print('\n Time for the mesh run : %f \n'%(time.time() - start))
            

    plt.figure(figsize=(15,3))
    plt.subplot(141)
    plt.imshow(ic_check[0].sum(axis=2))
    plt.title('Initial Conditions')

    plt.subplot(142)
    plt.imshow(fin[0].sum(axis=2))
    plt.title('TensorFlow (single GPU)')
    plt.colorbar()

    plt.subplot(143)
    plt.imshow(fin_check[0].sum(axis=2))
    plt.title('Mesh TensorFlow')
    plt.colorbar()

    plt.subplot(144)
    plt.imshow((fin_check[0] - fin[0]).sum(axis=2))
    plt.title('Residuals')
    plt.colorbar()

    plt.savefig("comparison_mesh.png")

    exit(0)
 

    
##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
