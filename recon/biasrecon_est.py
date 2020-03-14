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

sys.path.append('./utils/')
import tools
from getbiasparams import getbias
import diagnostics as dg
import fnn
import standardrecon as srecon
import cswisefuncs as cswisef
##


cosmology=Planck15
np.random.seed(100)
tf.random.set_random_seed(200)
cscratch = "/global/cscratch1/sd/chmodi/flowpm/recon/"


tf.flags.DEFINE_integer("gpus_per_node", 8, "Number of GPU on each node")
tf.flags.DEFINE_integer("gpus_per_task", 1, "Number of GPU in each task")
tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")

tf.flags.DEFINE_integer("nc", 128, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 400, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")
tf.flags.DEFINE_bool("nbody", True, "Do nbody evolution")
tf.flags.DEFINE_string("fpath", "", "suffix for the folder name")

#pyramid flags
tf.flags.DEFINE_integer("dsample", 2, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 16, "halo size")

#mesh flags
tf.flags.DEFINE_integer("nx", 1, "# blocks along x")
tf.flags.DEFINE_integer("ny", 1, "# blocks along y")
tf.flags.DEFINE_string("mesh_shape", "row:1;col:1", "mesh shape")
#tf.flags.DEFINE_string("layout", "nx:b1", "layout rules")
tf.flags.DEFINE_string("output_file", "timeline", "Name of the output timeline file")

tf.flags.DEFINE_integer("niter", 100, "number of iterations per loop")

FLAGS = tf.flags.FLAGS

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
stages = np.linspace(a0, a, nsteps, endpoint=True)
fpath = FLAGS.fpath
print(fpath)
for ff in [fpath, fpath + '/figs/', fpath + '/reconmeshes/']:
    try: os.makedirs(ff)
    except Exception as e: print (e)

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




def recon_model(mesh, data, bparams, ipkerror, R0, x0, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
                        a0=FLAGS.a0, a=FLAGS.af, nsteps=FLAGS.nsteps, dtype=tf.float32):
    """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """

    b1, b2, bs2 = bparams
    kerror, perror = ipkerror[0].astype(np.float32), ipkerror[1].astype(np.float32)
    
    
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
    pke_dim = mtf.Dimension("epk", len(perror))
    pkerror = mtf.import_tf_tensor(mesh, perror.astype(npdtype), shape=[pke_dim])

    
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
    
   
    if x0 is None:
        fieldvar = mtf.get_variable(mesh, 'linear', part_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=1, seed=None))
    else:
        fieldvar = mtf.get_variable(mesh, 'linear', part_shape, initializer = tf.constant_initializer(x0))


    state = mtfpm.lpt_init_single(fieldvar, a0, kv_lr, halo_size, lr_shape, hr_shape, part_shape[1:], antialias=True,)
    final_state = mtfpm.nbody_single(state, stages, lr_shape, hr_shape, kv_lr, halo_size)
    

    # paint the field
    final_field = mtf.zeros(mesh, shape=part_shape)
    final_field = mcomp.cic_paint_fr(final_field, final_state, part_shape, hr_shape, halo_size, splittables, mesh)
    
    ##
    #Get the fields for bias
    hr_field = mcomp.fr_to_hr(final_field, hr_shape, halo_size, splittables, mesh)
    mstate = mpm.mtf_indices(hr_field.mesh, shape=part_shape[1:], dtype=tf.float32)
    X = mtf.einsum([mtf.ones(hr_field.mesh, [batch_dim]), mstate], output_shape=[batch_dim] + mstate.shape[:])
    k_dims_pr = [d.shape[0] for d in kv]
    k_dims_pr = [k_dims_pr[2], k_dims_pr[0], k_dims_pr[1]]
    tfnc, tfbs = cswisef.float_to_mtf(nc*1., mesh, scalar), cswisef.float_to_mtf(bs, mesh, scalar)

    #
    initc = fieldvar
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
    
    model = ed*b1 + ed2*b2 + es2*bs2    
    mtfdata = mtf.import_tf_tensor(mesh, tf.convert_to_tensor(data), shape=shape)
    diff = model - mtfdata
    
    # Get prior
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
    prior = mtf.reduce_sum(mtf.square(cpfield)) * bs**3 #* nc**3

    # Total loss
    cdiff = mesh_utils.r2c3d(diff, k_dims_pr, dtype=cdtype)
    def _cwise_diff(kfield, pk, kx, ky, kz):
        kx = tf.reshape(kx, [-1, 1, 1])
        ky = tf.reshape(ky, [1, -1, 1])
        kz = tf.reshape(kz, [1, 1, -1])
        kk = tf.sqrt((kx / bs * nc)**2 + (ky / bs * nc)**2 + (kz / bs * nc)**2)
        kshape = kk.shape
        kk = tf.reshape(kk, [-1])
        pkmesh = tfp.math.interp_regular_1d_grid(x=kk, x_ref_min=kerror.min(), x_ref_max=kerror.max(),
                                                 y_ref=pk)
        priormesh = tf.reshape(pkmesh, kshape)
        priormesh = tf.cast(priormesh**0.5, kfield.dtype)
        return kfield / priormesh 
    
    cdiff = mtf.cwise(_cwise_diff, [cdiff, pkerror] + kv, output_dtype=cdtype) 

    def _cwise_smooth(kfield, kx, ky, kz):
        kx = tf.reshape(kx, [-1, 1, 1])
        ky = tf.reshape(ky, [1, -1, 1])
        kz = tf.reshape(kz, [1, 1, -1])
        kk = (kx / bs * nc)**2 + (ky/ bs * nc)**2 + (kz/ bs * nc)**2
        wts = tf.cast(tf.exp(- kk* (R0*bs/nc)**2), kfield.dtype)
        return kfield * wts

    cdiff = mtf.cwise(_cwise_smooth, [cdiff] + kv, output_dtype=cdtype)
    diff = mesh_utils.c2r3d(cdiff, diff.shape[-3:], dtype=dtype)
    chisq = mtf.reduce_sum(mtf.square(diff))
    loss = chisq + prior
    
    fields = [fieldvar, final_field, model]
    metrics = [chisq, prior, loss]
    
    return fields, metrics, kv





def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""

    #tf.logging.info("features = %s labels = %s mode = %s params=%s" %
    #              (features, labels, mode, params))

    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    fields, metrics, kv = recon_model(mesh, features['datasm'], features['bparams'], features['ipkerror'], features['R0'],  features['x0'])
    fieldvar, final, model = fields
    chisq, prior, loss = metrics
    
    ##
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    mesh_size = mesh_shape.size
    layout_rules = [("nx_lr", "row"), ("ny_lr", "col"),
                    ("nx", "row"), ("ny", "col"),
                    ("ty", "row"), ("tz", "col"),
                    ("ty_lr", "row"), ("tz_lr", "col"),
                    ("nx_block","row"), ("ny_block","col")]
    devices = ["/job:localhost/replica:0/task:%d/device:GPU:%d"%(i,j) for i in range(0) for j in range(FLAGS.gpus_per_task)]
    devices = [""] * mesh_size
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(mesh_shape, layout_rules, devices)
    print(mesh_impl)

    ##
    if mode == tf.estimator.ModeKeys.TRAIN:
        var_grads = mtf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables])
        optimizer = mtf.optimize.AdamWeightDecayOptimizer(features['lr'])
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)


    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    restore_hook = mtf.MtfRestoreHook(lowering)
    #
    tf_init = lowering.export_to_tf_tensor(fieldvar)
    tf_final = lowering.export_to_tf_tensor(final)
    tf_model = lowering.export_to_tf_tensor(model)
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_chisq = lowering.export_to_tf_tensor(chisq)
    tf_prior = lowering.export_to_tf_tensor(prior)
    
    ##Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf_loss = lowering.export_to_tf_tensor(loss)
        tf.summary.scalar("loss", tf_loss)
        tf.summary.scalar("chisq", tf_chisq)
        tf.summary.scalar("prior", tf_prior)
        predictions = {
            "ic": tf_init,
            "final": tf_final,
            "model": tf_model,
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            prediction_hooks=[restore_hook],
            export_outputs={
                "model": tf.estimator.export.PredictOutput(predictions) #TODO: is classify a keyword?
            })

    ##Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        tf_update_ops.append(tf.assign_add(global_step, 1))
        train_op = tf.group(tf_update_ops)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=1,
            keep_checkpoint_every_n_hours=2,
            defer_build=False, save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            fpath,
            save_steps=1000,
            saver=saver,
            listeners=[saver_listener])
        
        logging_hook = tf.train.LoggingTensorHook({"loss" : tf_loss, 
                                                   "chisq" : tf_chisq,
                                                   "prior" : tf_prior}, every_n_iter=10)

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(tf_loss, "loss")
        tf.identity(tf_prior, "prior")
        tf.identity(tf_chisq, "chisq")
        
        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar("loss", tf_loss)
        tf.summary.scalar("chisq", tf_chisq)
        tf.summary.scalar("prior", tf_prior)

        # restore_hook must come before saver_hook
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
            training_chief_hooks=[restore_hook, saver_hook, logging_hook])

    ##Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=tf_loss,
            evaluation_hooks=[restore_hook],
            eval_metric_ops={
                "loss": tf_loss,
                "chisq" : tf_chisq,
                #tf.metrics.accuracy(
                #    labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
            })
    



##############################################

def main(_):

    infield = True
    dtype=tf.float32
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    nc, bs = FLAGS.nc, FLAGS.box_size
    a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
    stages = np.linspace(a0, a, nsteps, endpoint=True)
    numd = 1e-3
    
    ##Begin here
    klin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../flowpm/data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)

    #pypath = '/global/cscratch1/sd/chmodi/cosmo4d/output/version2/L0400_N0128_05step-fof/lhd_S0100/n10/opt_s999_iM12-sm3v25off/meshes/'
    final = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/d/')
    ic = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/s/')
    fpos = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/dynamic/1/Position/')
    
    hpos = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/PeakPosition//')[1:int(bs**3 *numd)]
    hmass = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/Mass//')[1:int(bs**3 *numd)].flatten()

    meshpos = tools.paintcic(hpos, bs, nc)
    meshmass = tools.paintcic(hpos, bs, nc, hmass.flatten()*1e10)
    data = meshmass
    data /= data.mean()
    data -= 1
    kv = tools.fftk([nc, nc, nc], bs, symmetric=True, dtype=np.float32)
    datasm = tools.fingauss(data, kv, 3, np.pi*nc/bs)
    ic, data = np.expand_dims(ic, 0), np.expand_dims(data, 0).astype(np.float32)
    datasm = np.expand_dims(datasm, 0).astype(np.float32)
    print("Min in data : %0.4e"%datasm.min())
    
    np.save(fpath + 'ic', ic)
    np.save(fpath + 'data', data)

    ####################################################
    #
    tf.reset_default_graph()
    tfic = tf.constant(ic.astype(np.float32))
    state = lpt_init(tfic, a0=0.1, order=1)
    final_state = nbody(state,  stages, FLAGS.nc)
    tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])
    with tf.Session() as sess:
        state  = sess.run(final_state)
        
    fpos = state[0, 0]*bs/nc
    bparams, bmodel = getbias(bs, nc, data[0]+1, ic[0], fpos)
    #bmodel += 1 #np.expand_dims(bmodel, 0) + 1
    errormesh = data - np.expand_dims(bmodel, 0)
    kerror, perror = tools.power(errormesh[0]+1, boxsize=bs)
    kerror, perror = kerror[1:], perror[1:]
    print(kerror, perror)
    print("\nkerror", kerror.min(), kerror.max(), "\n")
    print("\nperror", perror.min(), perror.max(), "\n")
    suff = "-error"
    dg.saveimfig(suff, [ic, errormesh], [ic, data], fpath + '/figs/' )
    dg.save2ptfig(suff, [ic, errormesh], [ic, data], fpath + '/figs/', bs)
    ipkerror = iuspline(kerror, perror)

    ####################################################

    stdinit = srecon.standardinit(bs, nc, meshpos, hpos, final, R=8)

    recon_estimator = tf.estimator.Estimator(
      model_fn=model_fn,
        model_dir=fpath)

    
    def predict_input_fn(data=data, M0=0., w=3., R0=0., off=None, istd=None, x0=None):
        features = {}
        features['datasm'] = data
        features['R0'] = R0    
        features['x0'] = x0
        features['bparams'] = bparams
        features['ipkerror'] = [kerror, perror]#ipkerror
        return features, None
    
    eval_results = recon_estimator.predict(input_fn=lambda : predict_input_fn(x0 = ic), yield_single_examples=False)
    
    for i, pred in enumerate(eval_results):
        if i>0:     break
        
    suff = '-model'
    dg.saveimfig(suff, [pred['ic'], pred['model']], [ic, data], fpath + '/figs/' )
    dg.save2ptfig(suff, [pred['ic'], pred['model']], [ic, data], fpath + '/figs/', bs)
    np.save(fpath + '/reconmeshes/ic_true'+suff, pred['ic'])
    np.save(fpath + '/reconmeshes/fin_true'+suff, pred['final'])
    np.save(fpath + '/reconmeshes/model_true'+suff, pred['model'])

    #
    randominit = np.random.normal(size=data.size).reshape(data.shape)
    #eval_results = recon_estimator.predict(input_fn=lambda : predict_input_fn(x0 = np.expand_dims(stdinit, 0)), yield_single_examples=False)
    eval_results = recon_estimator.predict(input_fn=lambda : predict_input_fn(x0 = randominit), yield_single_examples=False)
    
    for i, pred in enumerate(eval_results):
        if i>0:     break
        
    suff = '-init'
    dg.saveimfig(suff, [pred['ic'], pred['model']], [ic, data], fpath + '/figs/' )
    dg.save2ptfig(suff, [pred['ic'], pred['model']], [ic, data], fpath + '/figs/', bs)
    np.save(fpath + '/reconmeshes/ic_init'+suff, pred['ic'])
    np.save(fpath + '/reconmeshes/fin_init'+suff, pred['final'])
    np.save(fpath + '/reconmeshes/model_init'+suff, pred['model'])

    #
    # Train and evaluate model.
    RRs = [4., 2., 1., 0.5, 0.]
    niter = 100
    iiter = 0

    for R0 in RRs:

        
        print('\nFor iteration %d\n'%iiter)
        print('With  R0=%0.2f \n'%(R0))

        def train_input_fn():
            features = {}
            features['datasm'] = data
            features['R0'] = R0    
            features['bparams'] = bparams
            features['ipkerror'] = [kerror, perror]#ipkerror
            #features['x0'] = np.expand_dims(stdinit, 0)
            features['x0'] = randominit 
            features['lr'] = 0.01
            return features, None

        recon_estimator.train(input_fn=train_input_fn, max_steps=iiter+niter)
        eval_results = recon_estimator.predict(input_fn=predict_input_fn, yield_single_examples=False)

        for i, pred in enumerate(eval_results):
            if i>0:     break

        iiter += niter#
        suff = '-%d-R%d'%(iiter, R0)
        dg.saveimfig(suff, [pred['ic'], pred['model']], [ic, data], fpath + '/figs/' )
        dg.save2ptfig(suff, [pred['ic'], pred['model']], [ic, data], fpath + '/figs/', bs)
        np.save(fpath + '/reconmeshes/ic'+suff, pred['ic'])
        np.save(fpath + '/reconmeshes/fin'+suff, pred['final'])
        np.save(fpath + '/reconmeshes/model'+suff, pred['model'])
                
    sys.exit(0)


    
##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
