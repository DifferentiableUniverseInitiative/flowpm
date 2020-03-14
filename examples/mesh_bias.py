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
import flowpm.mesh_composite as mcomp
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint

sys.path.append('../recon/utils/')
import tools
import diagnostics as dg
import cswisefuncs as cswisef
from getbiasparams import getbias
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


def shear(rfield, cfield, kv, tfnc, tfbs):

    wtss = mtf.slicewise(cswisef.shearwts, [cfield] + kv + [tfnc, tfbs],
                         output_shape=[cfield.shape]*6,
                         output_dtype=[cfield.dtype]*6,
                         splittable_dims=cfield.shape[:])
    
    counter = 0
    for i in range(3):
        for j in range(i, 3):
            baser = mesh_utils.c2r3d(wtss[counter], rfield.shape[-3:], dtype=rfield.dtype)
            rfield = rfield + baser*baser
            if i != j: rfield = rfield + baser*baser
            counter +=1
    return rfield




def nbody_prototype(mesh, bparams, infield=False, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
                        a0=FLAGS.a0, a=FLAGS.af, nsteps=FLAGS.nsteps, dtype=tf.float32):
    """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """
    b1, b2, bs2 = bparams
    
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
    splittables = lr_shape[:-1]+hr_shape[1:4]+part_shape[1:3]

#
    # Begin simulation
    
   
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

    print("\n final state : ", final_state)
    # paint the field
    final_field = mtf.zeros(mesh, shape=part_shape)
    final_field = mcomp.cic_paint_fr(final_field, final_state, output_shape=part_shape, hr_shape=hr_shape, halo_size=halo_size, splittables=splittables, mesh=mesh, weights=None)

    #Get the fields for bias
    hr_field = mcomp.fr_to_hr(final_field, hr_shape, halo_size, splittables, mesh)
    mstate = mpm.mtf_indices(hr_field.mesh, shape=part_shape[1:], dtype=tf.float32)
    X = mtf.einsum([mtf.ones(hr_field.mesh, [batch_dim]), mstate], output_shape=[batch_dim] + mstate.shape[:])
    k_dims_pr = [d.shape[0] for d in kv]
    k_dims_pr = [k_dims_pr[2], k_dims_pr[0], k_dims_pr[1]]
    tfnc, tfbs = cswisef.float_to_mtf(nc*1., mesh, scalar), cswisef.float_to_mtf(bs, mesh, scalar)

    #
    d0 = initc - mtf.reduce_mean(initc)
    #
    d2 = initc * initc
    d2  = d2  - mtf.reduce_mean(d2)
    #
    cfield = mesh_utils.r2c3d(d0, k_dims_pr, dtype=cdtype)
    shearfield = mtf.zeros(mesh, shape=part_shape)
    shearfield = shear(shearfield, cfield, kv, tfnc, tfbs)
    s2 = shearfield - mtf.reduce_mean(shearfield)
  
    dread = mcomp.cic_readout_fr(d0, [X], hr_shape=hr_shape, halo_size=halo_size, splittables=splittables, mesh=mesh)
    d2read = mcomp.cic_readout_fr(d2, [X], hr_shape=hr_shape, halo_size=halo_size, splittables=splittables, mesh=mesh)
    s2read = mcomp.cic_readout_fr(s2, [X], hr_shape=hr_shape, halo_size=halo_size, splittables=splittables, mesh=mesh)

    ed, ed2, es2 = mtf.zeros(mesh, shape=part_shape), mtf.zeros(mesh, shape=part_shape), mtf.zeros(mesh, shape=part_shape)
    ed = mcomp.cic_paint_fr(ed, final_state, output_shape=part_shape, hr_shape=hr_shape, halo_size=halo_size, splittables=splittables, mesh=mesh, weights=dread)
    ed2 = mcomp.cic_paint_fr(ed2, final_state, output_shape=part_shape, hr_shape=hr_shape, halo_size=halo_size, splittables=splittables, mesh=mesh, weights=d2read) 
    es2 = mcomp.cic_paint_fr(es2, final_state, output_shape=part_shape, hr_shape=hr_shape, halo_size=halo_size, splittables=splittables, mesh=mesh, weights=s2read)

    print(ed, ed2, es2)
    
    biasfield = ed*b1 + ed2*b2 + es2*bs2

    return initc, biasfield, input_field # 




##############################################

def main(_):

    infield = True
    dtype=tf.float32
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    mesh_size = mesh_shape.size
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
    devices = [""]*mesh_size
    print("List of devices", devices)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, devices)
    
    ##Begin here
    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)

    final = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/d/')
    ic = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/s/')
    fpos = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/dynamic/1/Position/')
    #ic = final
    finaltf = np.expand_dims(final, 0)

    pypath = '/global/cscratch1/sd/chmodi/cosmo4d/output/version2/L0400_N0128_05step-fof/lhd_S0100/n10/opt_s999_iM12-sm3v25off/meshes/'
    fin = tools.readbigfile(pypath + 'decic//') 
    
    hpos = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/PeakPosition//')[1:int(bs**3 *numd)]
    hmass = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/Mass//')[1:int(bs**3 *numd)].flatten()

    #meshpos = tools.paintcic(hpos, bs, nc)
    meshmass = tools.paintcic(hpos, bs, nc, hmass.flatten()*1e10)
    meshmass /= meshmass.mean()
    fin = meshmass

    #
    tf.reset_default_graph()
    tfic = tf.constant(np.expand_dims(ic, 0).astype(np.float32))
    state = lpt_init(tfic, a0=0.1, order=1)
    final_state = nbody(state,  stages, FLAGS.nc)
    tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])
    with tf.Session(server.target) as sess:
        state  = sess.run(final_state)
        
    fpos = state[0, 0]*bs/nc
    bparams, bmodel = getbias(bs, nc, meshmass, ic, fpos)
    #fin = bmodel
    bmodel = np.expand_dims(bmodel, 0) + 1
    
    fin = meshmass 
    ic, fin = np.expand_dims(ic, 0), np.expand_dims(fin, 0)
    print(ic.shape)
    
    tf.reset_default_graph()
    print('ic constructed')

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    
    initial_conditions, final_field, input_field = nbody_prototype(mesh, bparams, infield, nc=FLAGS.nc,
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

        print(fin_check.shape, fin.shape)
        #fin +=1
        fin_check +=1
        dg.saveimfig('-check', [ic_check, fin_check], [ic, fin], './tmp/')
        dg.save2ptfig('-check', [ic_check, fin_check], [ic, fin], './tmp/', bs)
        dg.saveimfig('-bmodel', [ic_check, bmodel], [ic, fin], './tmp/')
        dg.save2ptfig('-bmodel', [ic_check, bmodel], [ic, fin], './tmp/', bs)

##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
