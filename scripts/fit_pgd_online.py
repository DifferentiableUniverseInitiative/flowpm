import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys, os
sys.path.append('../flowpm/')
import flowpm
import pickle
from absl import app
from absl import flags
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from flowpm.tfpower import linear_matter_power
from flowpm import tfpm 
import jax_cosmo as jc
from flowpm.pk import pk as pkl
from flowpm.pk import _initialize_pk as initpk
import jax_cosmo.power as power
from flowpm.utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout


flags.DEFINE_string("filename", "results_fit_PGD.pkl", "Output filename")
flags.DEFINE_float("box_size", 64.,
                   "Transverse comoving size of the simulation volume")
flags.DEFINE_integer("nc", 64,
                     "Number of transverse voxels in the simulation volume")  
flags.DEFINE_integer("nsteps", 20, "Number of steps in the N-body simulation")
flags.DEFINE_integer("B", 1, "Scale resolution factor")


FLAGS = flags.FLAGS

#Run the Nbody
@tf.function
def nbody(cosmo,state,stages,nc,pm_nc_factor):
    stages=tf.cast(stages, dtype=tf.float32)
    states = tfpm.nbody(cosmo,
                           state,
                    stages, [nc,nc,nc],
                    pm_nc_factor,
                    return_intermediate_states=True)
    return states

def main(_):
    
    cosmology = flowpm.cosmology.Planck15()
    fftk = flowpm.kernels.fftk
    laplace_kernel = flowpm.kernels.laplace_kernel
    gradient_kernel = flowpm.kernels.gradient_kernel
    ia = -1 #Scale factor to fit for
    niters = 20
    # Create some initial conditions
    klin = tf.constant(np.logspace(-4, 1, 512), dtype=tf.float32)
    pk = linear_matter_power(cosmology, klin)
    pk_fun = lambda x: tf.cast(tf.reshape(interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), klin, pk), x.shape), tf.complex64)
    initial_conditions = flowpm.linear_field(
          [FLAGS.nc, FLAGS.nc, FLAGS.nc],
          [FLAGS.box_size, FLAGS.box_size, FLAGS.box_size],
          pk_fun,
          batch_size=1)
    initial_state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

    stages = np.linspace(0.1, 1., FLAGS.nsteps, endpoint=True)
    print(stages)
    states = nbody(cosmology,
                            initial_state,
                        stages, FLAGS.nc,
                        FLAGS.B
                        )
    print('Fiducial Simulation done')
    fpath = './fits/L%04d_N%04d_T%02d_B%d/'%(FLAGS.box_size, FLAGS.nc, FLAGS.nsteps, FLAGS.B)
    os.makedirs(fpath, exist_ok=True)
    np.save(fpath + 'fidstates', [states[i][1] for i in range(len(states))])
    pk_array=[]
    for i in range(len(states)):
        print(i, states[i][0])
        final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), states[i][1][0])
        final_field = tf.reshape(final_field, [FLAGS.nc, FLAGS.nc, FLAGS.nc])
        k, power_spectrum1 = pkl(final_field,shape=final_field.shape,boxsize=np.array([FLAGS.box_size, FLAGS.box_size,
                                             FLAGS.box_size]),kmin=np.pi/FLAGS.box_size,dk=2*np.pi/FLAGS.box_size)
        pk_array.append(power_spectrum1)

    np.save(fpath + 'fidps', pk_array)
    pk_jax={}
    cosmo=jc.Planck15()
    for i in range(len(states)):
        print(states[i][0], stages[i+1])
        pk_jax[stages[i+1]] = power.nonlinear_matter_power(cosmo, k, states[i][0])
    print('PS JAx computed computed')
    np.save(fpath + 'refps', [pk_jax[i] for i in pk_jax.keys()])
    # Run the Nbody with PGD fitting
    kl, ks = 0.3, 12
    alpha = 0.3 
    params0 = np.array([alpha, kl, ks]).astype(np.float32)
    pgdparams = tf.Variable(name='pgd', shape=(3), dtype=tf.float32,
                         initial_value=params0, trainable=True)
    statespgd, dmstate, params = tfpm.nbody_pgd(cosmology,
                                initial_state,
                                stages,  [FLAGS.nc, FLAGS.nc, FLAGS.nc],
                                FLAGS.box_size, 
                                pk_jax,
                                pgdparams, 
                                pm_nc_factor= FLAGS.B,
                                niters=niters,
                                return_intermediate_states=True)


    np.save(fpath + 'params', params)
    np.save(fpath + 'dmstate', dmstate)
    np.save(fpath + 'pgdstates', [statespgd[i][1] for i in range(len(states))])


if __name__ == "__main__":
  app.run(main)

