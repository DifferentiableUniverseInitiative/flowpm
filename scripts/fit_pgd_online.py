import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import sys, os
sys.path.append('../flowpm/')
sys.path.append('../../../DifferentiableHOS/')
sys.path.append('../../DifferentiableHOS/')
# import DifferentiableHOS as DHOS
import flowpm
import pickle
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from flowpm.tfpower import linear_matter_power
# import jax
from flowpm import tfpm_pgd as tfpm
import jax_cosmo as jc
from DifferentiableHOS.pk import pk as pkl
from DifferentiableHOS.pk import _initialize_pk as initpk
import jax_cosmo.power as power

cosmology = flowpm.cosmology.Planck15()

from flowpm.utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout
fftk = flowpm.kernels.fftk
laplace_kernel = flowpm.kernels.laplace_kernel
gradient_kernel = flowpm.kernels.gradient_kernel


box_size = 200.
nc = 128
nsteps= 20
B=1
pm_nc_factor = B
ia = -1 #Scale factor to fit for
niters = 20
fpath = './fits/L%04d_N%04d_T%02d_B%d/'%(box_size, nc, nsteps, B)
os.makedirs(fpath, exist_ok=True)

# Create some initial conditions
klin = tf.constant(np.logspace(-4, 1, 512), dtype=tf.float32)
pk = linear_matter_power(cosmology, klin)
pk_fun = lambda x: tf.cast(tf.reshape(interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), klin, pk), x.shape), tf.complex64)
initial_conditions = flowpm.linear_field(
      [nc, nc, nc],
      [box_size, box_size, box_size],
      pk_fun,
      batch_size=1)
initial_state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)

stages = np.linspace(0.1, 1., nsteps, endpoint=True)
print(stages)

# Run the Nbody
@tf.function
def nbody():
    states2 = tfpm.nbody(cosmology,
                           initial_state,
                    stages, [nc, nc, nc],
                    pm_nc_factor= B,
                    return_intermediate_states=True)
    return states2
states2 = nbody()
print('Fiducial Simulation done')
np.save(fpath + 'fidstates', [states2[i][1] for i in range(len(states2))])

pk_array=[]
for i in range(len(states2)):
    print(i, states2[i][0])
    final_field1 = flowpm.cic_paint(tf.zeros_like(initial_conditions), states2[i][1][0])
    final_field1 = tf.reshape(final_field1, [nc, nc, nc])
    k, power_spectrum1 = pkl(final_field1,shape=final_field1.shape,boxsize=np.array([box_size, box_size,
                                         box_size]),kmin=np.pi/box_size,dk=2*np.pi/box_size)
    pk_array.append(power_spectrum1)

np.save(fpath + 'fidps', pk_array)

    
pk_jax={}
cosmo=jc.Planck15()
for i in range(len(states2)):
#for aa in stages[1:]:
    #print(aa)
    print(states2[i][0], stages[i+1])
    pk_jax[stages[i+1]] = power.nonlinear_matter_power(cosmo, k, states2[i][0])
    #pk_jax[stages[i+1]] = pk_array[i]
print('PS JAx computed computed')
np.save(fpath + 'refps', [pk_jax[i] for i in pk_jax.keys()])


# Run the Nbody with PGD fitting
kl, ks = 0.3, 12
alpha = 0.3 
params0 = np.array([alpha, kl, ks]).astype(np.float32)
pgdparams = tf.Variable(name='pgd', shape=(3), dtype=tf.float32,
                     initial_value=params0, trainable=True)

#@tf.function
states2pgd, dmstate, params = tfpm.nbody_pgd(cosmology,
                            initial_state,
                            stages,  [nc, nc, nc],
                            box_size, 
                            pk_jax,
                            pgdparams, 
                            pm_nc_factor= B,
                            niters=niters,
                            return_intermediate_states=True)


np.save(fpath + 'params', params)
np.save(fpath + 'dmstate', dmstate)
np.save(fpath + 'pgdstates', [states2pgd[i][1] for i in range(len(states2))])

