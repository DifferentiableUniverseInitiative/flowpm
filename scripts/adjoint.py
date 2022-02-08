
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import sys
from time import time
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
import tfpm
#import tfpm_adj as tfpm
import jax_cosmo as jc
#from DifferentiableHOS.pk import pk as pkl
#from DifferentiableHOS.pk import _initialize_pk as initpk
#import jax_cosmo.power as power

cosmology = flowpm.cosmology.Planck15()
cosmo = cosmology

from flowpm.utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout
fftk = flowpm.kernels.fftk
laplace_kernel = flowpm.kernels.laplace_kernel
gradient_kernel = flowpm.kernels.gradient_kernel


box_size = 100.
nc = 8
nsteps = 4
stages = np.linspace(0.1, 1., nsteps, endpoint=True)
B=1
pm_nc_factor = B
ia = -1 #Scale factor to fit for
print("\n FOR %d steps\n"%nsteps)

klin = tf.constant(np.logspace(-4, 1, 512), dtype=tf.float32)
pk = linear_matter_power(cosmology, klin)
pk_fun = lambda x: tf.cast(tf.reshape(interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), klin, pk), x.shape), tf.complex64)


#GenIC and data
ic = flowpm.linear_field(
    [nc, nc, nc],
    [box_size, box_size, box_size],
    pk_fun,
    batch_size=1)

data = tf.random.uniform(ic.shape)

##############################################
### First, simple forward model and gradients with backpro
@tf.function
def pmsim(initial_conditions):
    print('gen pm graph')
    #initial_state = tf.stop_gradient(flowpm.lpt_init(cosmology, initial_conditions, 0.1))
    # state = tf.stop_gradient(flowpm.nbody(cosmology,
    #                         initial_state,
    #                     stages, [nc, nc, nc],
    #                     pm_nc_factor= B))
    initial_state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)
    state = flowpm.nbody(cosmology,
                            initial_state,
                        stages, [nc, nc, nc],
                        pm_nc_factor= B)
    print('Fiducial Simulation done')
    return state


@tf.function
def lossfunc(x, x0):
  field = flowpm.cic_paint(tf.zeros_like(ic), x)
  l  = tf.reduce_sum((field - x0)**2)
  return l


@tf.function
def gradloss(x, x0):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = lossfunc(x, x0)
    grad = tape.gradient(loss, x)
    return grad


@tf.function
def get_grads(ic, data):
    print('gen grad graph')
    with tf.GradientTape() as tape:
        tape.watch(ic)
        state = pmsim(ic)
        loss = lossfunc(state[0], data)
    grad = tape.gradient(loss, ic)
    return loss, grad

_ = pmsim(ic*np.random.uniform())
l1, backpropgrad = get_grads(ic, data)
print("loss 1 : ", l1)

#start = time()
#for i in range(10): pmsim(ic*np.random.uniform())
#print("time for 10 forward : ", time() - start) 

#start = time()
#for i in range(10): get_grads(ic*np.random.uniform(), data)
#print("time for 10 grads : ", time() - start) 



##############################################
## first force calculation for jump starting
@tf.function
def gradicadj(ic, x0):
    print('gen adjoint graph')
    state = tf.stop_gradient(pmsim(ic))
    loss = lossfunc(state[0], data)
    print('Loss : ', loss)
    adjx, adjv = 0.*gradloss(state[0], data), -1.*gradloss(state[0], data)
    adj = tf.stop_gradient(tfpm.adjoint(cosmo, state, adjx, adjv, stages[::-1].astype(np.float32), [nc, nc, nc]))
    state, adjx, adjv = adj[:3], adj[3:4], adj[4:5]
    
    @tf.function
    def _gradic(ic, adjx, adjv):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(ic)
            state = flowpm.lpt_init(cosmology, ic, 0.1)
            x, v, f = tf.split(state, 3, 0)
        gradx = tape.gradient(x, ic, output_gradients=adjv)
        gradv = tape.gradient(v, ic, output_gradients=adjx)
        return gradx + gradv
    grad = _gradic(ic, -adjx, -adjv)
    return loss, grad, state

start = time()
l2, adjgrad, state = gradicadj(ic, data)
print("time for making graph and first run adjoint : ", time() - start) 
print("loss 2 ", l2)

start = time()
for i in range(2): gradicadj(ic*np.random.uniform(), data)
print("time for 2 grads adjoint : ", time() - start) 

print(adjgrad/backpropgrad)
print(np.allclose(adjgrad, backpropgrad, atol=1e-3))
