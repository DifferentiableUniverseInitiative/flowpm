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
##if FLAGS.nbody: fpath = cscratch + "nbody_%d_nx%d_ny%d_mesh%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
##else: fpath = cscratch + "lpt_%d_nx%d_ny%d_mesh%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
##print(fpath)
##for ff in [fpath, fpath + '/figs']:
##    try: os.makedirs(ff)
##    except Exception as e: print (e)
##

bs, nc = 400, 128
numd = 1e-3

def setupfnn():

    ppath = '/project/projectdirs/m3058/chmodi/cosmo4d/train/L0400_N0128_05step-n10/width_3/Wts_30_10_1/r1rf1/hlim-13_nreg-43_batch-5/'
    pwts, pbias = [], []
    # act = [lambda x: relu(x), lambda x: relu(x), lambda x: sigmoid(x)]
    act = [lambda x: relu(x), lambda x: relu(x), lambda x: relu(x)]

    for s in [0, 2, 4]:
        pwts.append(np.load(ppath + 'w%d.npy'%s))
        pbias.append(np.load(ppath + 'b%d.npy'%s))
    pmx = np.load(ppath + 'mx.npy')
    psx = np.load(ppath + 'sx.npy')


    mpath = '/project/projectdirs/m3058/chmodi/cosmo4d/train/L0400_N0128_05step-n10/width_3/Wts_30_10_1/r1rf1/hlim-13_nreg-43_batch-5/eluWts-10_5_1/blim-20_nreg-23_batch-100/'
    mwts, mbias = [], []
    # act = [lambda x: relu(x), lambda x: relu(x), lambda x: sigmoid(x)]
    act = [lambda x: elu(x), lambda x: elu(x), lambda x: linear(x)]

    for s in [0, 2, 4]:
        mwts.append(np.load(mpath + 'w%d.npy'%s))
        mbias.append(np.load(mpath + 'b%d.npy'%s))
    mmx = np.load(mpath + 'mx.npy')
    msx = np.load(mpath + 'sx.npy')
    mmy = np.load(mpath + 'my.npy')
    msy = np.load(mpath + 'sy.npy')

    size = 3
    kernel = np.zeros([size, size, size, 1, size**3])
    for i in range(size):
        for j in range(size):
            for k in range(size):
                kernel[i, j, k, 0, i*size**2+j*size+k] = 1

    return [pwts, pbias, pmx, psx],  [mwts, mbias, mmx, msx, mmy, msy], kernel

def tfwrap3D(image, padding=1):
    
    upper_pad = image[:, -padding:,:, :]
    lower_pad = image[:, :padding,:, :]
    
    partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)
    
    left_pad = partial_image[:, :,-padding:, :]
    right_pad = partial_image[:, :,:padding, :]
    
    partial_image = tf.concat([left_pad, partial_image, right_pad], axis=2)
    
    front_pad = partial_image[:, :,:, -padding:]
    back_pad = partial_image[:, :,:, :padding]
    
    padded_image = tf.concat([front_pad, partial_image, back_pad], axis=3)
    return padded_image

def sinc(x):
    x = x + 1e-3 #x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
    return tf.sin(np.pi*x) / x/np.pi

def float_to_mtf(x, mesh, scalar):
    return mtf.import_tf_tensor(mesh, tf.constant(x, shape=[1]), shape=[scalar])

def _cwise_gauss(kfield, R, kx, ky, kz):
    kx = tf.reshape(kx, [-1, 1, 1]) * nc/bs
    ky = tf.reshape(ky, [1, -1, 1]) * nc/bs
    kz = tf.reshape(kz, [1, 1, -1]) * nc/bs
    kk = tf.sqrt(kx**2 + ky**2 + kz**2)
    wts = tf.exp(-0.5 * R**2 * kk**2)
    return kfield * tf.cast(wts, kfield.dtype)

def _cwise_decic(kfield, kx, ky, kz):
    kx = tf.reshape(kx, [-1, 1, 1]) * nc/bs
    ky = tf.reshape(ky, [1, -1, 1]) * nc/bs
    kz = tf.reshape(kz, [1, 1, -1]) * nc/bs
    wts = sinc(kx*bs/(2*np.pi*nc)) *sinc(ky*bs/(2*np.pi*nc)) *sinc(kz*bs/(2*np.pi*nc))
    wts = tf.pow(wts, -2.)
    return kfield * tf.cast(wts, kfield.dtype)

def _cwise_fingauss(kfield, R, kx, ky, kz):
    kny = 1*np.pi*nc/bs
    kx = tf.reshape(kx, [-1, 1, 1]) * nc/bs
    ky = tf.reshape(ky, [1, -1, 1]) * nc/bs
    kz = tf.reshape(kz, [1, 1, -1]) * nc/bs
    kk = tf.sqrt((2*kny/np.pi*tf.sin(kx*np.pi/(2*kny)))**2 + (2*kny/np.pi*tf.sin(ky*np.pi/(2*kny)))**2 + (2*kny/np.pi*tf.sin(kz*np.pi/(2*kny)))**2)
    wts = tf.exp(-0.5 * R**2 * kk**2)
    return kfield * tf.cast(wts, kfield.dtype)




def nbody_prototype(mesh, infield=False, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
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

   
    input_field = tf.placeholder(dtype, [batch_size, nc, nc, nc])
    if infield:
        initc = mtf.import_tf_tensor(mesh, input_field, shape=part_shape)
    else:
        initc = mtfpm.linear_field(mesh, shape, bs, nc, pk, kv)       


    print("initc : ", initc)
    
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
    ##
    x = final_field
    
    k_dims = [d.shape[0] for d in kv]
    k_dims = [k_dims[2], k_dims[0], k_dims[1]]
    x1f = mesh_utils.r2c3d(x, k_dims, dtype=cdtype)
    x1f = mtf.cwise(_cwise_decic, [x1f] + kv, output_dtype=cdtype) 
    #x1d = mtf.cast(x1f, dtype)
    x1d = mesh_utils.c2r3d(x1f, x.shape[-3:], dtype=dtype)
    x1d = mtf.add(x1d,  -1.)

    
    x1f0 = mesh_utils.r2c3d(x1d, k_dims, dtype=cdtype)
    x1f = mtf.cwise(_cwise_fingauss, [x1f0, float_to_mtf(R1, mesh, scalar)] + kv, output_dtype=cdtype) 
    x1 = mesh_utils.c2r3d(x1f, x1d.shape[-3:], dtype=dtype)
    x2f = mtf.cwise(_cwise_fingauss, [x1f0, float_to_mtf(R2, mesh, scalar)] + kv, output_dtype=cdtype) 
    x2 = mesh_utils.c2r3d(x2f, x1d.shape[-3:], dtype=dtype)
    x12 = x1-x2


    ppars, mpars, kernel = setupfnn()
    pwts, pbias, pmx, psx = ppars
    mwts, mbias, mmx, msx, mmy, msy = mpars
    msy, mmy = msy[0], mmy[0]
    print("mmy : ", mmy)
    size = 3
    
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
        pmodel = tf.nn.sigmoid(3 * yy3)
        return pmodel[...,0]
    
    pmodel = mtf.slicewise(apply_pwts,
                        [x, x1, x12],
                        output_dtype=tf.float32,
                        output_shape=part_shape, # + [mtf.Dimension('c_dim', 81)],
                        name='apply_pwts',
                        splittable_dims=lr_shape[:-1]+hr_shape[1:4]+part_shape[1:3])
    
    toret = pmodel
    
    #return initc, toret, input_field

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

    toret = pmodel*mmodel
    
    return initc, toret, input_field





##############################################

def main(_):

    infield = True
    dtype=tf.float32
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

    final = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/d/')
    ic = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/s/')
    #ic = final
    finaltf = np.expand_dims(final, 0)

    pypath = '/global/cscratch1/sd/chmodi/cosmo4d/output/version2/L0400_N0128_05step-fof/lhd_S0100/n10/opt_s999_iM12-sm3v25off/meshes/'
    fin = tools.readbigfile(pypath + 'decic//') 
    
    hpos = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/PeakPosition//')[1:int(bs**3 *numd)]
    hmass = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/Mass//')[1:int(bs**3 *numd)].flatten()

    #meshpos = tools.paintcic(hpos, bs, nc)
    meshmass = tools.paintcic(hpos, bs, nc, hmass.flatten()*1e10)
    fin = meshmass
    
    ic, fin = np.expand_dims(ic, 0), np.expand_dims(fin, 0)

    
    tf.reset_default_graph()
    print('ic constructed')

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    
    initial_conditions, final_field, input_field = nbody_prototype(mesh, infield, nc=FLAGS.nc,
                                                                       batch_size=FLAGS.batch_size, dtype=dtype)

    # Lower mesh computation
    
    start = time.time()
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    restore_hook = mtf.MtfRestoreHook(lowering)
    end = time.time()
    print('\n Time for lowering : %f \n'%(end - start))

    
    tf_initc = lowering.export_to_tf_tensor(initial_conditions)
    tf_final = lowering.export_to_tf_tensor(final_field)
    n_block_x, n_block_y, n_block_z = FLAGS.nx, FLAGS.ny, 1
    #nc = FLAGS.nc

    with tf.Session(server.target) as sess:
                    
        start = time.time()
        if infield:
            ic_check, fin_check = sess.run([tf_initc, tf_final], feed_dict={input_field:ic})
        else:
            ic_check, fin_check = sess.run([tf_initc, tf_final])
            ic, fin = ic_check, fin_check    
        print('\n Time for the mesh thingy : %f \n'%(time.time() - start))
        print(fin.sum())
        print(fin_check.sum())
              
        #fin +=1
        #fin_check +=1
        
        dg.saveimfig('-check', [ic_check, fin_check], [ic, fin], './tmp/')
        dg.save2ptfig('-check', [ic_check, fin_check], [ic, fin], './tmp/', bs)

    ppars, mpars, kernel = setupfnn()
    pwts, pbias, pmx, psx = ppars
    mwts, mbias, mmx, msx, mmy, msy = mpars
    print("mmy : ", mmy)
    
##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
