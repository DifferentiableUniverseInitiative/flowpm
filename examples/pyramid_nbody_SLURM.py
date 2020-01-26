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
cosmology=Planck15

tf.flags.DEFINE_integer("gpus_per_node", 8, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 2, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")

tf.flags.DEFINE_integer("nc", 64, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")

FLAGS = tf.flags.FLAGS


def lpt_prototype(nc=64, bs=200, batch_size=8, a0=0.1, a=1.0, nsteps=5, nproc=2):
    """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """
    # Compute a few things first, using simple tensorflow
    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)
    stages = np.linspace(a0, a, nsteps, endpoint=True)

    #pt = PerturbationGrowth(cosmology, a=[a], a_normalize=1.0)
    # Generate a batch of 3D initial conditions
    initial_conditions = flowpm.linear_field(nc,          # size of the cube
                                           bs,         # Physical size of the cube
                                           ipklin,      # Initial power spectrum
                                           batch_size=batch_size)

    state = lpt_init(initial_conditions, a0=a0, order=1) 
    final_state = nbody(state,  stages, nc)
    tfinal_field = cic_paint(tf.zeros_like(initial_conditions), final_state[0])

    ### Ok, now we implement the same thing but using Mesh TensorFlow ###
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    # Compute necessary Fourier kernels
    kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False)
    from flowpm.kernels import laplace_kernel, gradient_kernel
    lap = tf.cast(laplace_kernel(kvec), tf.complex64)
    grad_x = gradient_kernel(kvec, 0)
    grad_y = gradient_kernel(kvec, 1)
    grad_z = gradient_kernel(kvec, 2)


    # Define the named dimensions
    # Parameters of the small scales decomposition
    n_block_x = 2
    n_block_y = 1
    n_block_z = 1
    halo_size = 32

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

    nx_dim = mtf.Dimension('nx_block', n_block_x)
    ny_dim = mtf.Dimension('ny_block', n_block_y)
    nz_dim = mtf.Dimension('nz_block', n_block_z)

    sx_dim = mtf.Dimension('sx_block', nc//n_block_x)
    sy_dim = mtf.Dimension('sy_block', nc//n_block_y)
    sz_dim = mtf.Dimension('sz_block', nc//n_block_z)

    batch_dim = mtf.Dimension("batch", batch_size)
    pk_dim = mtf.Dimension("npk", len(plin))
    pk = mtf.import_tf_tensor(mesh, plin.astype('float32'), shape=[pk_dim])


    # kvec for low resolution grid
    kvec_lr = flowpm.kernels.fftk([lnc, lnc, lnc], symmetric=False) 

    kx_lr = mtf.import_tf_tensor(mesh, kvec_lr[0].squeeze().astype('float32')/ 2**downsampling_factor, shape=[x_dim])
    ky_lr = mtf.import_tf_tensor(mesh, kvec_lr[1].squeeze().astype('float32')/ 2**downsampling_factor, shape=[y_dim])
    kz_lr = mtf.import_tf_tensor(mesh, kvec_lr[2].squeeze().astype('float32')/ 2**downsampling_factor, shape=[z_dim])
    kv_lr = [kx_lr, ky_lr, kz_lr]

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


    initc = tf.reshape(initial_conditions, [1, 2, 1, 1, nc//2, nc, nc])

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
    low = mtf.reshape(low, lr_shape)

    state = mtfpm.lpt_init(low, high, 0.1, kv_lr, kv_hr, halo_size, hr_shape, lr_shape, part_shape[1:],
                            downsampling_factor=downsampling_factor, antialias=True,)

    # Here we can run our nbody
    final_state = mtfpm.nbody(state, stages, lr_shape, hr_shape, kv_lr, kv_hr, halo_size, downsampling_factor=downsampling_factor)

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

    final_field = mtf.reshape(final_field,  [batch_dim, fx_dim, fy_dim, fz_dim])

    return initial_conditions, tfinal_field, final_field
  
    ##

    
def main(_):
    num_tasks = int(os.environ['SLURM_NTASKS'])
    print('num_tasks : ', num_tasks)
        
    # Resolve the cluster from SLURM environment
    cluster = tf.distribute.cluster_resolver.SlurmClusterResolver({"mesh": num_tasks},
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
    #devices = ["/job:mesh/replica:0/task:%d/device:XLA_GPU:%d"%(i,j) for i in range(cluster_spec.num_tasks("mesh")) for j in range(FLAGS.gpus_per_task)]

    print("List of devices", devices)

    # Let's have a look!
    mesh_shape = [("row", 2), ("col", 1)]
    layout_rules = [("nx_lr", "row"), ("ny_lr", "col"),
                    ("nx", "row"), ("ny", "col"),
                    ("nx_block","row"), ("ny_block","col")]

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, devices)
    
    
    # Create computational graphs
    initial_conditions, final_field, mesh_final_field = lpt_prototype(nc=FLAGS.nc,
                                                                    batch_size=FLAGS.batch_size,
                                                                    nproc=len(devices))
    # Lower mesh computation
    graph = mesh_final_field.graph
    mesh = mesh_final_field.mesh
    lowering = mtf.Lowering(graph, {mesh:mesh_impl})

    # Retrieve output of computation
    result = lowering.export_to_tf_tensor(mesh_final_field)

    with tf.Session(server.target, config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)) as sess:
        a,b,c = sess.run([initial_conditions, final_field, result])
    np.save('init', a)
    np.save('reference_final', b)
    np.save('mesh_pyramid', c)
    
    
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

    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)
