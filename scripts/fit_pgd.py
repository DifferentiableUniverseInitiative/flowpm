import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../flowpm/')
sys.path.append('../../../DifferentiableHOS/')
sys.path.append('../../DifferentiableHOS/')
# import DifferentiableHOS as DHOS
import flowpm
import pickle
import flowpm.tfpower as tfpower
import flowpm.scipy.interpolate as interpolate
from flowpm.tfpower import linear_matter_power
# import jax
from flowpm import tfpm
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
nsteps= 10
B=1
pm_nc_factor = B
ia = -1 #Scale factor to fit for
niters = 50

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
# Run the Nbody
states2 = flowpm.nbody(cosmology,
                        initial_state,
                    stages, [nc, nc, nc],
                    pm_nc_factor= B,
                    return_intermediate_states=True)

print('Simulation done')

#Nyquist and number of modes in case we need weight for PGD fitting loss
kny = np.pi*nc/box_size
Nmodes = initpk(shape=initial_state.shape[1:],boxsize=np.array([box_size, box_size,
                                            box_size]),kmin=np.pi/box_size,dk=2*np.pi/box_size)[1].numpy().real[1:-1]


#######################################
#######################################
##Estimate PS for the last time snapshot
pk_array=[]
pk_array1=[]

# for i in range(len(states2)):
for i in [ia]:
    print(i)
    final_field1 = flowpm.cic_paint(tf.zeros_like(initial_conditions), states2[i][1][0])
    final_field1 = tf.reshape(final_field1, [nc, nc, nc])
    k, power_spectrum1 = pkl(final_field1,shape=final_field1.shape,boxsize=np.array([box_size, box_size,
                                         box_size]),kmin=np.pi/box_size,dk=2*np.pi/box_size)
    pk_array1.append(power_spectrum1)

    
pk_jax=[]
cosmo=jc.Planck15()
# for i in range(len(states2)):
for i in [ia]:
    print(i)
    pk_jax.append(power.nonlinear_matter_power(cosmo, k, states2[i][0]))

print(len(k))
print(len(pk_jax[-1]))
print('PS computed')

#######################################
#######################################
#Same correction function as flowpm but writing as tf.function
@tf.function
def PGD_correction(
        state,
        pgdparams):
    """                                                                                                                                                                                                                                                                                                                          """
    print('new graph')
    state = tf.convert_to_tensor(state, name="state")

    shape = state.get_shape()
    batch_size = shape[1]
    ncpm = [nc, nc, nc]
    ncf = [nc * pm_nc_factor]*3
#     ncf = [n * pm_nc_factor for n in nc]

    rho = tf.zeros([batch_size] + ncf)
    wts = tf.ones((batch_size, ncpm[0] * ncpm[1] * ncpm[2]))
    nbar = ncpm[0] * ncpm[1] * ncpm[2] / (ncf[0] * ncf[1] * ncf[2])

    rho = cic_paint(rho, tf.multiply(state[0], pm_nc_factor), wts)
    rho = tf.multiply(rho,
                      1. / nbar)  # I am not sure why this is not needed here                                                                                                                                                                                                                                                 
    delta_k = r2c3d(rho, norm=ncf[0] * ncf[1] * ncf[2])
    alpha, kl, ks = tf.split(pgdparams, 3)
    update = tfpm.apply_PGD(tf.multiply(state[0], pm_nc_factor),
                             delta_k,
                             alpha,
                              kl,
                              ks,
                               )
    update = tf.expand_dims(update, axis=0) / pm_nc_factor
    return update


@tf.function
def pgd(params):
#     alpha, kl, ks = tf.split(params, 3)
    print('pgd graph')
    dx = PGD_correction(states2[ia][1], params)
    new_state= states2[ia][1][0]+dx
    final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), new_state)
    final_field=tf.reshape(final_field, [nc, nc, nc])
    k, power_spectrum = pkl(final_field,shape=final_field.shape,boxsize=np.array([box_size, box_size,
                                            box_size]),kmin=np.pi/box_size,dk=2*np.pi/box_size)

    return k, power_spectrum, final_field


#######################################
#######################################
#Finally fit
#Loss 
#weight = Nmodes**0.5  * (1-k/kny)
weight = 1-k/kny
weight = weight/weight.sum()
weight = tf.cast(weight, tf.float32)
# weight = 1.

@tf.function
def get_grads(params):
    with tf.GradientTape() as tape:
        tape.watch(params)
        k, ps, field = pgd(params)
#         loss = tf.reduce_sum(ps)
        psref = tf.constant(np.array(pk_jax[ia]))
        loss = tf.reduce_sum((weight*(1 - ps/psref))**2)
    grad = tape.gradient(loss, params)
    return loss, grad


kl, ks = 0.3, 12
alpha = 0.3 
params0 = np.array([alpha, kl, ks]).astype(np.float32)
params = tf.Variable(name='pgd', shape=(3), dtype=tf.float32,
                             initial_value=params0, trainable=True)
print('Init params : ', params)

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
losses, pparams = [], []
print(params0)
for i in range(niters):
    loss, grads = get_grads(params)
    opt.apply_gradients(zip([grads], [params]))
    print(i, loss, params)
    losses.append(loss)
    pparams.append(params.numpy().copy())

print('Final params : ', params)
#Plot fit params
plt.plot(losses)
plt.semilogy()
plt.grid(which='both')
plt.savefig('losses.png')
plt.close()


k, ps, field = pgd(params)
k, ps0, field0 = pgd(tf.constant(params0))
plt.plot(k, pk_array1[ia]/pk_jax[ia], "k:", label='B=1')
plt.plot(k, ps0/pk_jax[ia], 'k--', label='0')
plt.semilogx()
plt.grid(which='both')
# plt.fill_between(k, k*0 + 1 + 1/Nmodes**0.5, k*0 + 1 - 1/Nmodes**0.5)
for i in range(niters/10.):
    j = 10*i + 9
    try:
        k, ps, field = pgd(tf.constant(pparams[j]))
        plt.plot(k, ps/pk_jax[ia], '-', label=j)
    except: pass
plt.legend()
plt.savefig('pgdfit.png')
