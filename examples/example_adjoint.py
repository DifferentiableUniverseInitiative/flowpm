import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import sys
from time import time

sys.path.append("../flowpm/")
sys.path.append("../../../DifferentiableHOS/")
sys.path.append("../../DifferentiableHOS/")
import flowpm
import flowpm.scipy.interpolate as interpolate
from flowpm.tfpower import linear_matter_power
import tfpm
import jax_cosmo as jc

cosmology = flowpm.cosmology.Planck15()
cosmo = cosmology

from flowpm.utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout

fftk = flowpm.kernels.fftk
laplace_kernel = flowpm.kernels.laplace_kernel
gradient_kernel = flowpm.kernels.gradient_kernel

box_size = 100.0
nc = 8
nsteps = 4
stages = np.linspace(0.1, 1.0, nsteps, endpoint=True)
B = 1
pm_nc_factor = B
print("\n FOR %d steps\n" % nsteps)

klin = tf.constant(np.logspace(-4, 1, 512), dtype=tf.float32)
pk = linear_matter_power(cosmology, klin)
pk_fun = lambda x: tf.cast(
    tf.reshape(
        interpolate.interp_tf(tf.reshape(tf.cast(x, tf.float32), [-1]), klin, pk),
        x.shape,
    ),
    tf.complex64,
)


# GenIC and data
ic = flowpm.linear_field(
    [nc, nc, nc], [box_size, box_size, box_size], pk_fun, batch_size=1
)

data = tf.random.uniform(ic.shape)

##############################################
### First, simple forward model and gradients with backpro
@tf.function
def pmsim(initial_conditions):
    print("gen pm graph")
    initial_state = flowpm.lpt_init(cosmology, initial_conditions, 0.1)
    state = flowpm.nbody(cosmology, initial_state, stages, [nc, nc, nc], pm_nc_factor=B)
    return state


@tf.function
def lossfunc(x, x0):
    field = flowpm.cic_paint(tf.zeros_like(ic), x)
    l = tf.reduce_sum((field - x0) ** 2)
    return l


@tf.function
def gradloss(x, x0):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = lossfunc(x, x0)
    grad = tape.gradient(loss, x)
    return grad


@tf.function
def backprop_grads(ic, data):
    print("gen grad graph")
    with tf.GradientTape() as tape:
        tape.watch(ic)
        state = pmsim(ic)
        loss = lossfunc(state[0], data)
    grad = tape.gradient(loss, ic)
    return loss, grad


_ = pmsim(ic * np.random.uniform())
l1, backpropgrad = backprop_grads(ic, data)
print("loss 1 : ", l1)


##############################################
## first force calculation for jump starting


@tf.function
def gradicadj(ic, x0):
    print("gen adjoint graph")
    state = tf.stop_gradient(pmsim(ic))
    loss = lossfunc(state[0], data)
    print("Loss : ", loss)
    adjx, adjv = 0.0 * gradloss(state[0], data), -1.0 * gradloss(state[0], data)
    adj = tf.stop_gradient(
        tfpm.adjoint(
            cosmo, state, adjx, adjv, stages[::-1].astype(np.float32), [nc, nc, nc]
        )
    )
    state, adjx, adjv = adj[:3], adj[3:4], adj[4:5]
    grad = tfpm.adjoint_lptinit(cosmo, ic, -adjx, -adjv, a0=0.1)
    return loss, grad, state


start = time()
l2, adjgrad, state = gradicadj(ic, data)
print("time for making graph and first run adjoint : ", time() - start)
print("loss 2 ", l2)

# print(adjgrad/backpropgrad)
print(
    "adjoint gradients and backprop are close : ",
    np.allclose(adjgrad, backpropgrad, atol=1e-3),
)

##############################################
#####Time testing
niters = 10
start = time()
for i in range(niters):
    pmsim(ic * np.random.uniform())
print("time for %d forward : " % niters, time() - start)

start = time()
for i in range(niters):
    backprop_grads(ic * np.random.uniform(), data)
print("time for %d grads : " % niters, time() - start)

start = time()
for i in range(niters):
    gradicadj(ic * np.random.uniform(), data)
print("time for %d grads adjoint : " % niters, time() - start)
