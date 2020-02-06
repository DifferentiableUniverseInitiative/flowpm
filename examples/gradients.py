import numpy as np
import os
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mesh_tensorflow as mtf
import flowpm.mesh_ops as mpm
import flowpm.mtfpm as mtfpm
import flowpm.mesh_utils as mesh_utils
import flowpm
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt
import time

cosmology=Planck15

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
tf.flags.DEFINE_integer("dsample", 2, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 32, "halo size")

#mesh flags
tf.flags.DEFINE_integer("nx", 4, "# blocks along x")
tf.flags.DEFINE_integer("ny", 2, "# blocks along y")
tf.flags.DEFINE_string("mesh_shape", "row:16", "mesh shape")
#tf.flags.DEFINE_string("layout", "nx:b1", "layout rules")
tf.flags.DEFINE_string("output_file", "timeline", "Name of the output timeline file")

FLAGS = tf.flags.FLAGS


def gradients_prototype(nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
                        a0=FLAGS.a0, a=FLAGS.af, nsteps=FLAGS.nsteps, dtype=tf.float32):
    """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """
    if dtype == tf.float32: npdtype = "float32"
    elif dtype == tf.float64: npdtype = "float64"
    print(dtype, npdtype)
    
    # Compute a few things first, using simple tensorflow
    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    stages = np.linspace(a0, a, nsteps, endpoint=True)

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    # Define the named dimensions
    # Parameters of the small scales decomposition
    n_block_x = FLAGS.nx
    n_block_y = FLAGS.ny
    n_block_z = 1
    halo_size = FLAGS.hsize

    if halo_size >= 0.5*min(nc//n_block_x, nc//n_block_y, nc//n_block_z):
        new_size = int(0.5*min(nc//n_block_x, nc//n_block_y, nc//n_block_z))
        print('WARNING: REDUCING HALO SIZE from %d to %d'%(halo_size, new_size))
        halo_size = new_size
        

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
    pk = mtf.import_tf_tensor(mesh, plin.astype(npdtype), shape=[pk_dim])

    # Compute necessary Fourier kernels
    kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False, dtype=npdtype)
    kx = mtf.import_tf_tensor(mesh, kvec[0].squeeze().astype(npdtype), shape=[tfx_dim])
    ky = mtf.import_tf_tensor(mesh, kvec[1].squeeze().astype(npdtype), shape=[tfy_dim])
    kz = mtf.import_tf_tensor(mesh, kvec[2].squeeze().astype(npdtype), shape=[tfz_dim])
    kv = [ky, kz, kx]
    print(kv)
    
    # kvec for low resolution grid
    kvec_lr = flowpm.kernels.fftk([lnc, lnc, lnc], symmetric=False, dtype=npdtype)

    kx_lr = mtf.import_tf_tensor(mesh, kvec_lr[0].squeeze().astype(npdtype)/ 2**downsampling_factor, shape=[tx_dim])
    ky_lr = mtf.import_tf_tensor(mesh, kvec_lr[1].squeeze().astype(npdtype)/ 2**downsampling_factor, shape=[ty_dim])
    kz_lr = mtf.import_tf_tensor(mesh, kvec_lr[2].squeeze().astype(npdtype)/ 2**downsampling_factor, shape=[tz_dim])
    kv_lr = [ky_lr, kz_lr, kx_lr]

    # kvec for high resolution blocks
    padded_sx_dim = mtf.Dimension('padded_sx_block', nc//n_block_x+2*halo_size)
    padded_sy_dim = mtf.Dimension('padded_sy_block', nc//n_block_y+2*halo_size)
    padded_sz_dim = mtf.Dimension('padded_sz_block', nc//n_block_z+2*halo_size)

    kvec_hr = flowpm.kernels.fftk([nc//n_block_x+2*halo_size, nc//n_block_y+2*halo_size, nc//n_block_z+2*halo_size], symmetric=False, dtype=npdtype)
    kx_hr = mtf.import_tf_tensor(mesh, kvec_hr[0].squeeze().astype(npdtype), shape=[padded_sx_dim])
    ky_hr = mtf.import_tf_tensor(mesh, kvec_hr[1].squeeze().astype(npdtype), shape=[padded_sy_dim])
    kz_hr = mtf.import_tf_tensor(mesh, kvec_hr[2].squeeze().astype(npdtype), shape=[padded_sz_dim])
    kv_hr = [ky_hr, kz_hr, kx_hr]

    shape = [batch_dim, fx_dim, fy_dim, fz_dim]
    lr_shape = [batch_dim, x_dim, y_dim, z_dim]
    hr_shape = [batch_dim, nx_dim, ny_dim, nz_dim, sx_dim, sy_dim, sz_dim]
    part_shape = [batch_dim, fx_dim, fy_dim, fz_dim]

    # Compute initial initial conditions distributed
    initc = mtfpm.linear_field(mesh, shape, bs, nc, pk, kv, dtype=dtype)

    # Reshaping array into high resolution mesh
    field = mtf.slicewise(lambda x:tf.expand_dims(tf.expand_dims(tf.expand_dims(x, axis=1),axis=1),axis=1),
                      [initc],
                      output_dtype=initc.dtype,
                      output_shape=hr_shape,
                      name='my_reshape',
                      splittable_dims=lr_shape[:-1]+hr_shape[1:4]+part_shape[1:3])

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
                        output_dtype=initc.dtype,
                        output_shape=lr_shape,
                        name='my_dumb_reshape',
                        splittable_dims=lr_shape[:-1]+hr_shape[:4])


    state = mtfpm.lpt_init(low, high, 0.1, kv_lr, kv_hr, halo_size, hr_shape, lr_shape,
                           part_shape[1:], downsampling_factor=downsampling_factor, antialias=True, dtype=dtype)

    # Here we can run our nbody
    final_state = state #mtfpm.nbody(state, stages, lr_shape, hr_shape, kv_lr, kv_hr, halo_size, downsampling_factor=downsampling_factor, dtype=dtype)

    # paint the field
    final_field = mtf.zeros(mesh, shape=hr_shape)
    for block_size_dim in hr_shape[-3:]:
        final_field = mtf.pad(final_field, [halo_size, halo_size], block_size_dim.name)
    final_field = mesh_utils.cic_paint(final_field, final_state[0], halo_size, dtype=dtype)
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

    return initc, final_field, stages


def main(_):

    dtype=tf.float32

    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)

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

    
    # Create computational graphs
    initial_conditions, mesh_final_field, stages = gradients_prototype(nc=FLAGS.nc,
                                                                       batch_size=FLAGS.batch_size, dtype=dtype)
    mtfgrads = mtf.gradients([mtf.reduce_sum(mtf.multiply(mesh_final_field, mesh_final_field))], [initial_conditions])
    # Lower mesh computation
    graph = mesh_final_field.graph
    mesh = mesh_final_field.mesh

    start = time.time()
    lowering = mtf.Lowering(graph, {mesh:mesh_impl})
    end = time.time()
    print('\n Time for lowering : %f \n'%(end - start))

    # Retrieve output of computation
    result = lowering.export_to_tf_tensor(mesh_final_field)
    initc  = lowering.export_to_tf_tensor(initial_conditions)
    grads  = lowering.export_to_tf_tensor(mtfgrads[0])


    # Run normal flowpm
    state = lpt_init(initc, a0=0.1, order=1, dtype=dtype)
    final_state = state # nbody(state,  stages, FLAGS.nc, dtype=dtype)
    tfinal_field = cic_paint(tf.zeros_like(initc), final_state[0])
    tfgrads = tf.gradients(tf.reduce_sum(tfinal_field**2), initc)

    with tf.Session(server.target) as sess:
        a,b,c = sess.run([initc, tfinal_field, result])
        gb,gc = sess.run([tfgrads, grads])
    gb = gb[0]
    np.save('init', a)
    np.save('reference_final', b)
    np.save('mesh_pyramid', c)
    np.save('gradsmesh', gc)
    np.save('gradstf', gb)

    ##
    plt.figure(figsize=(15,3))
    plt.subplot(141)
    plt.imshow(a[0].sum(axis=2))
    plt.title('Initial Conditions')

    plt.subplot(142)
    plt.imshow(b[0].sum(axis=2))
    plt.title('TensorFlow (single GPU)')
    plt.colorbar()

    plt.subplot(143)
    plt.imshow(c[0].sum(axis=2))
    plt.title('Mesh TensorFlow')
    plt.colorbar()

    plt.subplot(144)
    plt.imshow((b[0] - c[0]).sum(axis=2))
    plt.title('Residuals')
    plt.colorbar()

    plt.savefig("comparison.png")

    ##
    vmax = None
    vmin = None #-1*vmax
    plt.figure(figsize=(8,3))
    plt.subplot(131)
    plt.imshow(gb[0].sum(axis=2), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('single GPU')

    plt.subplot(132)
    plt.imshow(gc[0].sum(axis=2), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Mesh')

    plt.subplot(133)
    plt.imshow((gb-gc)[0].sum(axis=2), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Residual')

    if dtype == tf.float32: plt.savefig("gradientscomparison.png")
    elif dtype == tf.float64: plt.savefig("gradientscomparison-64.png")
            
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)
