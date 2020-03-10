import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import mesh_tensorflow as mtf
print(mtf)

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
if FLAGS.nbody: fpath = cscratch + "nbody_%d_nx%d_ny%d_mesh%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
else: fpath = cscratch + "lpt_%d_nx%d_ny%d_mesh%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
print(fpath)
for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)


def recon_model(mesh, data, R0, x0, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
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



    # Begin simulation
    
   
    if x0 is None:
        fieldvar = mtf.get_variable(mesh, 'linear', part_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=1, seed=None))
    else:
        #fieldvar = mtf.get_variable(mesh, 'linear', part_shape, initializer = tf.constant_initializer(tf.convert_to_tensor(x0)))
        fieldvar = mtf.get_variable(mesh, 'linear', part_shape, initializer = tf.constant_initializer(x0))
    print("\nfieldvar : \n", fieldvar)
    


    # Here we can run our nbody
    if FLAGS.nbody:
        state = mtfpm.lpt_init_single(fieldvar, a0, kv_lr, halo_size, lr_shape, hr_shape, part_shape[1:], antialias=True,)
        # Here we can run our nbody
        final_state = mtfpm.nbody_single(state, stages, lr_shape, hr_shape, kv_lr, halo_size)
    else:
        final_state = mtfpm.lpt_init_single(fieldvar, stages[-1], kv_lr, halo_size, lr_shape, hr_shape, part_shape[1:], antialias=True,)

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
    prior = mtf.reduce_sum(mtf.square(cpfield)) * bs**3 #*nc**3

    # Total loss
    diff = (final_field - mtfdata)
    R0 = tf.constant(R0)
    print("R0 in the recon_model : ", R0)
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
    
    fields = [fieldvar, final_field]
    metrics = [chisq, prior, loss]
    
    return fields, metrics, kv




##############################################



def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""

    #tf.logging.info("features = %s labels = %s mode = %s params=%s" %
    #              (features, labels, mode, params))

    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    data = features['data']
    R0 = features['R0']*1.
    x0 = features['x0']
    print("\nR0 in the model function : %0.1f\n"%R0)
    fields, metrics, kv = recon_model(mesh, data, R0, x0)
    fieldvar, final_field = fields
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

#        nyq = np.pi*nc/bs
#        def _cwise_highpass(kfield, kx, ky, kz):
#            kx = tf.reshape(kx, [-1, 1, 1])
#            ky = tf.reshape(ky, [1, -1, 1])
#            kz = tf.reshape(kz, [1, 1, -1])
#            kk = (kx / bs * nc)**2 + (ky/ bs * nc)**2 + (kz/ bs * nc)**2
#            wts = tf.cast(tf.exp(- kk* (R0*bs/nc + 1/nyq)**2), kfield.dtype)
#            return kfield * (1-wts)
#        
#        k_dims_pr = [d.shape[0] for d in kv]
#        k_dims_pr = [k_dims_pr[2], k_dims_pr[0], k_dims_pr[1]]
#        cgrads = mesh_utils.r2c3d(var_grads[0], k_dims_pr, dtype=tf.complex64)
#        cgrads = mtf.cwise(_cwise_highpass, [cgrads] + kv, output_dtype=tf.complex64)
#        var_grads = [mesh_utils.c2r3d(cgrads, var_grads[0].shape[-3:], dtype=tf.float32)]
#        update_ops = [mtf.assign(fieldvar, fieldvar - var_grads[0]*0.2)]

        #optimizer = mtf.optimize.AdafactorOptimizer(1)
        #optimizer = mtf.optimize.SgdOptimizer(0.01)
        #optimizer = mtf.optimize.MomentumOptimizer(0.01, 0.001)
        optimizer = mtf.optimize.AdamWeightDecayOptimizer(features['lr'])
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

#

    start = time.time()
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    print("\nTime taken for lowering is : %0.3f"%(time.time()-start))
    restore_hook = mtf.MtfRestoreHook(lowering)
    #
    tf_init = lowering.export_to_tf_tensor(fieldvar)
    tf_data = lowering.export_to_tf_tensor(final_field)
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
            "data": tf_data,
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            prediction_hooks=[restore_hook],
            export_outputs={
                "data": tf.estimator.export.PredictOutput(predictions) #TODO: is classify a keyword?
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

    dtype=tf.float32
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)

    startw = time.time()
    
    print(mesh_shape)
##
##    
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
        with tf.Session() as sess:
            ic, fin  = sess.run([tfic, tfinal_field])
        np.save(fpath + 'ic', ic)
        np.save(fpath + 'final', fin)

        
    print(ic.shape, fin.shape)
    ########################################################
    print(ic.shape, fin.shape)
    recon_estimator = tf.estimator.Estimator(
      model_fn=model_fn,
        model_dir=fpath)

    def eval_input_fn():
        features = {}
        features['data'] = fin
        features['R0'] = 0
        features['x0'] = None
        features['lr'] = 0
        return features, None

    # Train and evaluate model.

    RRs = [4., 2., 1., 0.5, 0.]
    niter = 200
    iiter = 0

    for R0 in RRs:
        print('\nFor iteration %d and R=%0.1f\n'%(iiter, R0))

        def train_input_fn():
            features = {}
            features['data'] = fin
            features['R0'] = R0
            features['x0'] = np.random.normal(size=fin.size).reshape(fin.shape)
            features['lr'] = 0.01
            return features, None

        for _ in range(1):
            recon_estimator.train(input_fn=train_input_fn, max_steps=iiter+niter)
            eval_results = recon_estimator.predict(input_fn=eval_input_fn, yield_single_examples=False)
            
            for i, pred in enumerate(eval_results):
                if i>0:     break
            
            iiter += niter#
            dg.saveimfig(iiter, [pred['ic'], pred['data']], [ic, fin], fpath + '/figs/' )
            dg.save2ptfig(iiter, [pred['ic'], pred['data']], [ic, fin], fpath + '/figs/', bs)

    sys.exit(0)
##
##

if __name__ == "__main__":
  tf.app.run(main=main)

  
