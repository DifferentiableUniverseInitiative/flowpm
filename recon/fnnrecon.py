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
from fnn import *
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
tf.flags.DEFINE_float("box_size", 400, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")
tf.flags.DEFINE_bool("nbody", False, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "", "suffix for the folder name")

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

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
stages = np.linspace(a0, a, nsteps, endpoint=True)
fpath = cscratch + "fnn_nx%d_ny%d_mesh%s/"%(FLAGS.nx, FLAGS.ny, FLAGS.suffix)
print(fpath)
for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)

numd = 1e-3


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
    print("Dtype : ", dtype, npdtype)
    
    # Compute a few things first, using simple tensorflow
    kny = 1*np.pi*nc/bs
    R1, R2 = 3., 3*1.2
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

    scalar = mtf.Dimension("scalar", 1)    

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

    #k_dims = [tx_dim, ty_dim, tz_dim]

    batch_dim = mtf.Dimension("batch", batch_size)

    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)
    pk_dim = mtf.Dimension("npk", len(plin))
    pk = mtf.import_tf_tensor(mesh, plin.astype(npdtype), shape=[pk_dim])

    
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

#
    # Begin simulation
    
    ## Compute initial initial conditions distributed
    #initc = mtfpm.linear_field(mesh, shape, bs, nc, pk, kv)

   
    fieldvar = mtf.get_variable(mesh,'linear', part_shape)
    input_field = tf.placeholder(data.dtype, [batch_size, nc, nc, nc])
    mtfinp = mtf.import_tf_tensor(mesh, input_field, shape=part_shape)
    linearop = mtf.assign(fieldvar, mtfinp)

    #field = fieldvar
    initc = fieldvar



    print("initc : ", initc)
    
    # Here we can run our nbody
    if FLAGS.nbody:
        state = mtfpm.lpt_init_single(fieldvar, a0, kv_lr, halo_size, lr_shape, hr_shape, part_shape[1:], antialias=True,)
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
    ##
    x = final_field
    
    ppars, mpars, kernel = setupfnn()
    pwts, pbias, pmx, psx = ppars
    mwts, mbias, mmx, msx, mmy, msy = mpars
    msy, mmy = msy[0], mmy[0]
    print("mmy : ", mmy)
    size = 3
    

    k_dims = [d.shape[0] for d in kv]
    k_dims = [k_dims[2], k_dims[0], k_dims[1]]
    tfnc, tfbs = float_to_mtf(nc*1., mesh, scalar), float_to_mtf(bs, mesh, scalar)

    x1f = mesh_utils.r2c3d(x, k_dims, dtype=cdtype)
    x1f = mtf.cwise(cwise_decic, [x1f] + kv + [tfnc, tfbs], output_dtype=cdtype) 
    x1d = mesh_utils.c2r3d(x1f, x.shape[-3:], dtype=dtype)
    x1d = mtf.add(x1d,  -1.)

    
    x1f0 = mesh_utils.r2c3d(x1d, k_dims, dtype=cdtype)
    x1f = mtf.cwise(cwise_fingauss, [x1f0, float_to_mtf(R1, mesh, scalar)] + kv + [tfnc, tfbs], output_dtype=cdtype) 
    x1 = mesh_utils.c2r3d(x1f, x1d.shape[-3:], dtype=dtype)
    x2f = mtf.cwise(cwise_fingauss, [x1f0, float_to_mtf(R2, mesh, scalar)] + kv + [tfnc, tfbs], output_dtype=cdtype) 
    x2 = mesh_utils.c2r3d(x2f, x1d.shape[-3:], dtype=dtype)
    x12 = x1-x2

    width = tf.placeholder(tf.float32, shape=())

    def apply_pwts(x, x1, x2):
        #y = tf.expand_dims(x, axis=-1)
    
        y = tf.nn.conv3d(tf.expand_dims(x, axis=-1), kernel, [1, 1, 1, 1, 1], 'SAME')
        y1 = tf.nn.conv3d(tf.expand_dims(x1, axis=-1), kernel, [1, 1, 1, 1, 1], 'SAME')
        y2 = tf.nn.conv3d(tf.expand_dims(x2, axis=-1), kernel, [1, 1, 1, 1, 1], 'SAME')
        #y = tf.nn.conv3d(tf.expand_dims(tfwrap3D(x), -1), kernel, [1, 1, 1, 1, 1], 'VALID')
        #y1 = tf.nn.conv3d(tf.expand_dims(tfwrap3D(x1), -1), kernel, [1, 1, 1, 1, 1], 'VALID')
        #y2 = tf.nn.conv3d(tf.expand_dims(tfwrap3D(x12), -1), kernel, [1, 1, 1, 1, 1], 'VALID')

        yy = tf.concat([y, y1, y2], axis=-1)
        yy = yy - pmx
        yy = yy / psx
        yy1 = tf.nn.relu(tf.matmul(yy, pwts[0]) + pbias[0])
        yy2 = tf.nn.relu(tf.matmul(yy1, pwts[1]) + pbias[1])
        yy3 = tf.matmul(yy2, pwts[2]) + pbias[2]
        pmodel = tf.nn.sigmoid(width * yy3)
        return pmodel[...,0]
    
    pmodel = mtf.slicewise(apply_pwts,
                        [x, x1, x12],
                        output_dtype=tf.float32,
                        output_shape=part_shape, # + [mtf.Dimension('c_dim', 81)],
                        name='apply_pwts',
                        splittable_dims=lr_shape[:-1]+hr_shape[1:4]+part_shape[1:3])
    

    def apply_mwts(x, x1, x2):
        #y = tf.expand_dims(x, axis=-1)

        zz = tf.concat([tf.expand_dims(x, -1), tf.expand_dims(x1, -1), tf.expand_dims(x2, -1)], axis=-1)
        zz = zz - mmx
        zz = zz / msx
        zz1 = tf.nn.elu(tf.matmul(zz, mwts[0]) + mbias[0])
        zz2 = tf.nn.elu(tf.matmul(zz1, mwts[1]) + mbias[1])
        zz3 = tf.matmul(zz2, mwts[2]) + mbias[2]
        mmodel = zz3*msy + mmy
        return mmodel[...,0]
    
    mmodel = mtf.slicewise(apply_mwts,
                        [x, x1, x12],
                        output_dtype=tf.float32,
                        output_shape=part_shape, # + [mtf.Dimension('c_dim', 81)],
                        name='apply_mwts',
                        splittable_dims=lr_shape[:-1]+hr_shape[1:4]+part_shape[1:3])

    model = pmodel*mmodel
    
    mtfdata = mtf.import_tf_tensor(mesh, tf.convert_to_tensor(data), shape=shape)
    
    # Get prior
    #k_dims = [d.shape[0] for d in kv]
    #k_dims = [k_dims[2], k_dims[0], k_dims[1]]
    k_dims_pr = [d.shape[0] for d in kv]
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
    
    cpfield = mtf.cwise(_cwise_prior, [cfield, pk] + kv, output_dtype=tf.float32) 
    prior = mtf.reduce_sum(mtf.square(cpfield)) * bs**3 

    # Total loss
    #diff = (model - mtfdata)
    modelf = mesh_utils.r2c3d(model, k_dims, dtype=cdtype)
    modelsmf = mtf.cwise(cwise_fingauss, [modelf, float_to_mtf(R1, mesh, scalar)] + kv + [tfnc, tfbs], output_dtype=cdtype) 
    modelsm = mesh_utils.c2r3d(modelsmf, x1d.shape[-3:], dtype=dtype)
    dataf = mesh_utils.r2c3d(mtfdata, k_dims, dtype=cdtype)
    datasmf = mtf.cwise(cwise_fingauss, [dataf, float_to_mtf(R1, mesh, scalar)] + kv + [tfnc, tfbs], output_dtype=cdtype) 
    datasm = mesh_utils.c2r3d(datasmf, x1d.shape[-3:], dtype=dtype)

    ##Anneal
    R0 = tf.placeholder(tf.float32, shape=())
    M0 = tf.placeholder(tf.float32, shape=())
    diff = mtf.log(modelsm + M0) - mtf.log(datasm + M0)
    diff = diff / 0.25

    def _cwise_smooth(kfield, kx, ky, kz):
        kx = tf.reshape(kx, [-1, 1, 1])
        ky = tf.reshape(ky, [1, -1, 1])
        kz = tf.reshape(kz, [1, 1, -1])
        kk = (kx / bs * nc)**2 + (ky/ bs * nc)**2 + (kz/ bs * nc)**2
        wts = tf.cast(tf.exp(- kk* (R0*bs/nc)**2), kfield.dtype)
        return kfield * wts

    cdiff = mesh_utils.r2c3d(diff, k_dims_pr, dtype=cdtype)
    cdiff = mtf.cwise(_cwise_smooth, [cdiff] + kv, output_dtype=cdtype)
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
    cgrads = mtf.cwise(_cwise_highpass, [cgrads] + kv, output_dtype=cdtype)
    var_grads = [mesh_utils.c2r3d(cgrads, diff.shape[-3:], dtype=dtype)]

    lr = tf.placeholder(tf.float32, shape=())
    update_op = mtf.assign(fieldvar, fieldvar - var_grads[0]*lr)

    return initc, model, loss, var_grads, update_op, linearop, input_field, lr, R0, M0, width, chisq, prior





##############################################

def main(_):

    infield = True
    dtype=tf.float32
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    nc, bs = FLAGS.nc, FLAGS.box_size
    a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
    stages = np.linspace(a0, a, nsteps, endpoint=True)
    numd = 1e-3
    
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

    final = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/d/')
    ic = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/s/')

    pypath = '/global/cscratch1/sd/chmodi/cosmo4d/output/version2/L0400_N0128_05step-fof/lhd_S0100/n10/opt_s999_iM12-sm3v25off/meshes/'
    fin = tools.readbigfile(pypath + 'decic//') 
    
    hpos = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/PeakPosition//')[1:int(bs**3 *numd)]
    hmass = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/Mass//')[1:int(bs**3 *numd)].flatten()

    #meshpos = tools.paintcic(hpos, bs, nc)
    meshmass = tools.paintcic(hpos, bs, nc, hmass.flatten()*1e10)
    data = meshmass
    
    ic, data = np.expand_dims(ic, 0), np.expand_dims(data, 0).astype(np.float32)
    np.save(fpath + 'ic', ic)
    np.save(fpath + 'data', data)


    ####################################################
    tf.reset_default_graph()
    print('ic constructed')

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    
    initial_conditions, data_field, loss, var_grads, update_op, linear_op, input_field, lr, R0, M0, width, chisq, prior = recon_prototype(mesh, data, nc=FLAGS.nc, batch_size=FLAGS.batch_size, dtype=dtype)

    # Lower mesh computation
    
    start = time.time()
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    restore_hook = mtf.MtfRestoreHook(lowering)
    end = time.time()
    print('\n Time for lowering : %f \n'%(end - start))

    
    tf_initc = lowering.export_to_tf_tensor(initial_conditions)
    tf_data = lowering.export_to_tf_tensor(data_field)
    tf_chisq = lowering.export_to_tf_tensor(chisq)
    tf_prior = lowering.export_to_tf_tensor(prior)
    tf_grads = lowering.export_to_tf_tensor(var_grads[0])
    #tf_lr = lowering.export_to_tf_tensor(lr)
    tf_linear_op = lowering.lowered_operation(linear_op)
    tf_update_ops = lowering.lowered_operation(update_op)
    n_block_x, n_block_y, n_block_z = FLAGS.nx, FLAGS.ny, 1
    nc = FLAGS.nc


    with tf.Session(server.target) as sess:
                    
        start = time.time()
        sess.run(tf_linear_op, feed_dict={input_field:ic})
        ic_check, data_check = sess.run([tf_initc, tf_data], {width:3})
        
        dg.saveimfig('-check', [ic_check, data_check], [ic, data], fpath + '/figs/')
        dg.save2ptfig('-check', [ic_check, data_check], [ic, data], fpath + '/figs/', bs)
        print('Total time taken for mesh thingy is : ', time.time()-start)

        sess.run(tf_linear_op, feed_dict={input_field:np.random.normal(size=ic.size).reshape(ic.shape)})
        ic0, data0 = sess.run([tf_initc, tf_data], {width:3})
        dg.saveimfig('-init', [ic0, data0], [ic, data], fpath)
        start = time.time()

        titer = 20
        niter = 101
        iiter = 0
        
        start0 = time.time()
        RRs = [4, 2, 1, 0.5, 0]
        wws = [1, 2, 3]
        lrs = np.array([0.1, 0.1, 0.1, 0.1, 0.1])*2
        #lrs = [0.1, 0.05, 0.01, 0.005, 0.001]

        for mm in [1e12, 1e11]:
            for iR, zlR in enumerate(zip(RRs, lrs)):
                RR, lR = zlR
                for ww in [1, 2, 3]:
                    for ff in [fpath + '/figs-M%02d-R%02d-w%01d'%(np.log10(mm), 10*RR, ww)]:
                        try: os.makedirs(ff)
                        except Exception as e: print (e)

                    for i in range(niter):
                        iiter +=1
                        sess.run(tf_update_ops, {lr:lR, M0:mm, R0:RR, width:ww})
                        if (i%titer == 0):
                            end = time.time()
                            print('Iter : ', i)
                            print('Time taken for %d iterations: '%titer, end-start)
                            start = end

                            ##
                            ic1, data1, cc, pp = sess.run([tf_initc, tf_data, tf_chisq, tf_prior], {M0:mm, R0:RR, width:ww})
                            print('Chisq and prior are : ', cc, pp)

                            dg.saveimfig(i, [ic1, data1], [ic, data], ff)
                            dg.save2ptfig(i, [ic1, data1], [ic, data], ff, bs)
                    dg.saveimfig(iiter, [ic1, data1], [ic, data], fpath+'/figs')
                    dg.save2ptfig(iiter, [ic1, data1], [ic, data], fpath+'/figs', bs)
            wws = [3]
            RRs = [1, 0.5, 0]
        ic1, data1 = sess.run([tf_initc, tf_data], {width:3})
        print('Total time taken for %d iterations is : '%iiter, time.time()-start0)
        
    dg.saveimfig(i, [ic1, data1], [ic, data], fpath)
    dg.save2ptfig(i, [ic1, data1], [ic, data], fpath, bs)

    np.save(fpath + 'ic_recon', ic1)
    np.save(fpath + 'data_recon', data1)
    print('Total wallclock time is : ', time.time()-start0)
    
##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
