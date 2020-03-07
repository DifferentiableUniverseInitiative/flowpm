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

sys.path.append('./utils/')
import tools
import diagnostics as dg
##

cosmology=Planck15
np.random.seed(100)
tf.random.set_random_seed(200)
cscratch = "/global/cscratch1/sd/chmodi/flowpm/recon/"


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

#pyramid flags
tf.flags.DEFINE_integer("dsample", 2, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 32, "halo size")
tf.flags.DEFINE_string("suffix", "", "suffix for the folder name")

#mesh flags
tf.flags.DEFINE_integer("nx", 4, "# blocks along x")
tf.flags.DEFINE_integer("ny", 2, "# blocks along y")
tf.flags.DEFINE_string("mesh_shape", "row:16", "mesh shape")
#tf.flags.DEFINE_string("layout", "nx:b1", "layout rules")
tf.flags.DEFINE_string("output_file", "timeline", "Name of the output timeline file")

FLAGS = tf.flags.FLAGS

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
stages = np.linspace(a0, a, nsteps, endpoint=True)
if FLAGS.nbody: fpath = cscratch + "nbody_%d_nx%d_ny%d_pyramid%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
else: fpath = cscratch + "lpt_%d_nx%d_ny%d_pyramid%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)



def recon_prototype(mesh, data, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
                        a0=FLAGS.a0, a=FLAGS.af, nsteps=FLAGS.nsteps, dtype=tf.float32):
    """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """
    if dtype == tf.float32:
        npdtype = "float32"
        cdtype = tf.complex64
    elif dtype == tf.float64:
        npdtype = "float64"
        cdtype = tf.complex128
    print(dtype, npdtype)
    
    # Compute a few things first, using simple tensorflow
    stages = np.linspace(a0, a, nsteps, endpoint=True)

    #graph = mtf.Graph()
    #mesh = mtf.Mesh(graph, "my_mesh")

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

    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)
    pk_dim = mtf.Dimension("npk", len(plin))
    pk = mtf.import_tf_tensor(mesh, plin.astype(npdtype), shape=[pk_dim])

    # Compute necessary Fourier kernels
    kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False, dtype=npdtype)
    kx = mtf.import_tf_tensor(mesh, kvec[0].squeeze().astype(npdtype), shape=[tfx_dim])
    ky = mtf.import_tf_tensor(mesh, kvec[1].squeeze().astype(npdtype), shape=[tfy_dim])
    kz = mtf.import_tf_tensor(mesh, kvec[2].squeeze().astype(npdtype), shape=[tfz_dim])
    kv = [ky, kz, kx]
    
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


    # kvec for prior blocks
    prior_sx_dim = mtf.Dimension('prior_sx_block', nc//n_block_x)
    prior_sy_dim = mtf.Dimension('prior_sy_block', nc//n_block_y)
    prior_sz_dim = mtf.Dimension('prior_sz_block', nc//n_block_z)

    kvec_pr = flowpm.kernels.fftk([nc//n_block_x, nc//n_block_y, nc//n_block_z], symmetric=False, dtype=npdtype)
    kx_pr = mtf.import_tf_tensor(mesh, kvec_pr[0].squeeze().astype(npdtype), shape=[prior_sx_dim])
    ky_pr = mtf.import_tf_tensor(mesh, kvec_pr[1].squeeze().astype(npdtype), shape=[prior_sy_dim])
    kz_pr = mtf.import_tf_tensor(mesh, kvec_pr[2].squeeze().astype(npdtype), shape=[prior_sz_dim])
    kv_pr = [ky_pr, kz_pr, kx_pr]


    shape = [batch_dim, fx_dim, fy_dim, fz_dim]
    lr_shape = [batch_dim, x_dim, y_dim, z_dim]
    hr_shape = [batch_dim, nx_dim, ny_dim, nz_dim, sx_dim, sy_dim, sz_dim]
    part_shape = [batch_dim, fx_dim, fy_dim, fz_dim]

    ## Compute initial initial conditions distributed
    
    fieldvar = mtf.get_variable(mesh,'linear', hr_shape)
    input_field = tf.placeholder(data.dtype, [batch_size, n_block_x, n_block_y, n_block_z, nc//n_block_x, nc//n_block_y, nc//n_block_z])
    mtfinp = mtf.import_tf_tensor(mesh, input_field, shape=hr_shape)
    linearop = mtf.assign(fieldvar, mtfinp)
    #
    field = fieldvar
    initc = mtf.slicewise(lambda x: x[:,0,0,0],
                    [field],
                    output_dtype=tf.float32,
                    output_shape=[batch_dim, fx_dim, fy_dim, fz_dim],
                    name='my_dumb_reshape',
                    splittable_dims=part_shape[:-1]+hr_shape[:4])

    #
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

    # Here we can run our nbody
    if FLAGS.nbody:
        state = mtfpm.lpt_init(low, high, 0.1, kv_lr, kv_hr, halo_size, hr_shape, lr_shape,
                           part_shape[1:], downsampling_factor=downsampling_factor, antialias=True)

        final_state = mtfpm.nbody(state, stages, lr_shape, hr_shape, kv_lr, kv_hr, halo_size, downsampling_factor=downsampling_factor)
    else:
        final_state = mtfpm.lpt_init(low, high, stages[-1], kv_lr, kv_hr, halo_size, hr_shape, lr_shape,
                           part_shape[1:], downsampling_factor=downsampling_factor, antialias=True)


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

    mtfdata = mtf.import_tf_tensor(mesh, tf.convert_to_tensor(data), shape=shape)
    
    # Get prior
    k_dims_pr = [d.shape[0] for d in kv_pr]
    k_dims_pr = [k_dims_pr[2], k_dims_pr[0], k_dims_pr[1]]
    cfield = mesh_utils.r2c3d(fieldvar, k_dims_pr, dtype=cdtype)
    def _cwise_prior(kfield, pk, kx, ky, kz):
        kx = tf.reshape(kx, [-1, 1, 1])
        ky = tf.reshape(ky, [1, -1, 1])
        kz = tf.reshape(kz, [1, 1, -1])
        kk = tf.sqrt((kx / bs * nc)**2 + (ky / bs * nc)**2 + (kz / bs * nc)**2)
        kshape = kk.shape
        kk = tf.reshape(kk, [-1])
        pkmesh = tfp.math.interp_regular_1d_grid(x=kk, x_ref_min=1e-05, x_ref_max=1000.0,
                                                 y_ref=pk, grid_regularizing_transform=tf.log)
        priormesh = tf.reshape(pkmesh, kshape)
        return tf.abs(kfield) / priormesh**0.5 
    
    cpfield = mtf.cwise(_cwise_prior, [cfield, pk] + kv_pr, output_dtype=tf.float32) 
    prior = mtf.reduce_sum(mtf.square(cpfield)) * bs**3 
    
    # Total loss
    diff = (final_field - mtfdata)
    R0 = tf.placeholder(tf.float32, shape=())
    def _cwise_smooth(kfield, kx, ky, kz):
        kx = tf.reshape(kx, [-1, 1, 1])
        ky = tf.reshape(ky, [1, -1, 1])
        kz = tf.reshape(kz, [1, 1, -1])
        kk = (kx / bs * nc)**2 + (ky/ bs * nc)**2 + (kz/ bs * nc)**2
        wts = tf.cast(tf.exp(- kk* (R0*bs/nc)**2), kfield.dtype)
        return kfield * wts
    cdiff = mesh_utils.r2c3d(diff, k_dims_pr, dtype=cdtype)
    cdiff = mtf.cwise(_cwise_smooth, [cdiff] + kv_pr, output_dtype=cdtype)
    diff = mesh_utils.c2r3d(cdiff, diff.shape[-3:], dtype=dtype)
    chisq = mtf.reduce_sum(mtf.square(diff))
    loss = chisq + prior

    
    #return initc, final_field, loss, linearop, input_field
    nyq = np.pi*nc/bs
    def _cwise_highpass(kfield, kx, ky, kz):
        kx = tf.reshape(kx, [-1, 1, 1])
        ky = tf.reshape(ky, [1, -1, 1])
        kz = tf.reshape(kz, [1, 1, -1])
        kk = (kx / bs * nc)**2 + (ky/ bs * nc)**2 + (kz/ bs * nc)**2
        wts = tf.cast(tf.exp(- kk* (R0*bs/nc + 1/nyq)**2), kfield.dtype)
        return kfield * (1-wts)

    var_grads = mtf.gradients([loss], [fieldvar])
    cgrads = mesh_utils.r2c3d(var_grads[0], k_dims_pr, dtype=cdtype)
    cgrads = mtf.cwise(_cwise_highpass, [cgrads] + kv_pr, output_dtype=cdtype)
    var_grads = [mesh_utils.c2r3d(cgrads, var_grads[0].shape[-3:], dtype=dtype)]

    lr = tf.placeholder(tf.float32, shape=())
    update_op = mtf.assign(fieldvar, fieldvar - var_grads[0]*lr)

    return initc, final_field, loss, var_grads, update_op, linearop, input_field, lr, R0



##############################################

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
    
    ##Begin here
    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)

    tf.reset_default_graph()
    # Run normal flowpm to generate data
    try:
        ic, fin = np.load(fpath + 'ic.npy'), np.load(fpath + 'final.npy')
        print('Data loaded')
    except Exception as e:
        print('Exception occured', e)
        tfic = linear_field(FLAGS.nc, FLAGS.box_size, ipklin, batch_size=1, seed=100, dtype=dtype)
        if FLAGS.nbody:
            state = lpt_init(tfic, a0=0.1, order=1)
            final_state = nbody(state,  stages, FLAGS.nc)
        else:
            final_state = lpt_init(tfic, a0=stages[-1], order=1)
        tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])
        with tf.Session(server.target) as sess:
            ic, fin  = sess.run([tfic, tfinal_field])
        np.save(fpath + 'ic', ic)
        np.save(fpath + 'final', fin)

        
    tf.reset_default_graph()
    print('ic constructed')

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    
    initial_conditions, final_field, loss, var_grads, update_op, linear_op, input_field, lr, R0 = recon_prototype(mesh, fin, nc=FLAGS.nc,
                                                                       batch_size=FLAGS.batch_size, dtype=dtype)

    # Lower mesh computation

    start = time.time()
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    restore_hook = mtf.MtfRestoreHook(lowering)
    end = time.time()
    print('\n Time for lowering : %f \n'%(end - start))


    
    tf_initc = lowering.export_to_tf_tensor(initial_conditions)
    tf_final = lowering.export_to_tf_tensor(final_field)
    tf_grads = lowering.export_to_tf_tensor(var_grads[0])
    tf_linear_op = lowering.lowered_operation(linear_op)
    tf_update_ops = lowering.lowered_operation(update_op)
    n_block_x, n_block_y, n_block_z = FLAGS.nx, FLAGS.ny, 1
    nc = FLAGS.nc
    ic_hrshape = ic.reshape([FLAGS.batch_size,  n_block_x, nc//n_block_x, n_block_y, nc//n_block_y,  n_block_z, nc//n_block_z])
    ic_hrshape = np.transpose(ic_hrshape, [0, 1, 3, 5, 2, 4, 6])
    with tf.Session(server.target) as sess:
                    
        #ic_check, fin_check = sess.run([tf_initc, tf_final])
        sess.run(tf_linear_op, feed_dict={input_field:ic_hrshape})
        ic_check, fin_check = sess.run([tf_initc, tf_final])
        dg.saveimfig('-check', [ic_check, fin_check], [ic, fin], fpath)
        dg.save2ptfig('-check', [ic_check, fin_check], [ic, fin], fpath, bs)

        sess.run(tf_linear_op, feed_dict={input_field:np.random.normal(size=ic.size).reshape(ic_hrshape.shape)})
        ic0, fin0 = sess.run([tf_initc, tf_final])
        dg.saveimfig('-init', [ic0, fin0], [ic, fin], fpath)
        start = time.time()

        niter = 5
        iiter = 0
        start0 = time.time()
        RRs = [4, 2, 1, 0.5, 0]
        lrs = np.array([0.2, 0.15, 0.1, 0.1, 0.1])
        #lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
        
        for iR, zlR in enumerate(zip(RRs, lrs)):
            RR, lR = zlR
            #for ff in [fpath + '/figs-R%02d'%(10*RR)]:
            for ff in [fpath + '/figsiter']:
                try: os.makedirs(ff)
                except Exception as e: print (e)
            for i in range(301):
                if (i%niter == 0):
                    end = time.time()
                    print('Iter : ', i)
                    print('Time taken for %d iterations: '%niter, end-start)
                    start = end
                    ##
                    ic1, fin1 = sess.run([tf_initc, tf_final])
                    
                    #dg.saveimfig(i, [ic1, fin1], [ic, fin], fpath+'/figs-R%02d'%(10*RR))
                    #dg.save2ptfig(i, [ic1, fin1], [ic, fin], fpath+'/figs-R%02d'%(10*RR), bs)
                    dg.saveimfig2x2(iiter, [ic1, fin1], [ic, fin], fpath+'/figsiter')
                    #
                sess.run(tf_update_ops, {lr:lR, R0:RR})
                iiter +=1
                    
            dg.saveimfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs')
            dg.save2ptfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs', bs)

        ic1, fin1 = sess.run([tf_initc, tf_final])
        print('Total time taken for %d iterations is : '%iiter, time.time()-start0)
        
    dg.saveimfig(i, [ic1, fin1], [ic, fin], fpath)
    dg.save2ptfig(i, [ic1, fin1], [ic, fin], fpath, bs)

    np.save(fpath + 'ic_recon', ic1)
    np.save(fpath + 'final_recon', fin1)
    print('Total wallclock time is : ', time.time()-start0)

##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)
