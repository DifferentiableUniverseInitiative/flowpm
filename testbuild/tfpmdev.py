# Module storing a few tensorflow function to implement FastPM
import numpy as np
import numpy
import tensorflow as tf
from tfpmfuncs import *
from background import *

def rtflpt1(dlin_k, pos, config):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at q. The result has the same dtype as q.
    """
    bs, nc = config['boxsize'], config['nc']    
    #ones = tf.ones_like(dlin_k)
    lap = tflaplace(config)
    
    displacement = tf.zeros_like(pos)
    displacement = []
    for d in range(3):
        #kweight = tf.multiply(tfgradient(config, d), lap)
        kweight = tfgradient(config, d) * lap
        dispc = tf.multiply(kweight, dlin_k)
        disp = tf.multiply(tf.spectral.irfft3d(dispc), nc**3)
        displacement.append(cic_readout(disp, pos))

    return tf.stack(displacement, axis=1)


def tflpt1(dlin_k, pos, config):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at q. The result has the same dtype as q.
    """
    bs, nc = config['boxsize'], config['nc']    
    #ones = tf.ones_like(dlin_k)
    lap = tflaplace(config)
    
    displacement = tf.zeros_like(pos)
    displacement = []
    for d in range(3):
        #kweight = tf.multiply(tfgradient(config, d), lap)
        kweight = tfgradient(config, d) * lap
        dispc = tf.multiply(kweight, dlin_k)
        disp = tf.multiply(tf.cast(tf.spectral.ifft3d(dispc), tf.float32), nc**3)
        displacement.append(cic_readout(disp, pos))

    return tf.stack(displacement, axis=1)



def rtflpt2source(dlin_k, config):
    """ Generate the second order LPT source term.  """

    bs, nc = config['boxsize'], config['nc']
    source = tf.zeros((nc, nc, nc))
    
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []

    # diagnoal terms
    lap = tflaplace(config)
    
    for d in range(3):
        grad = tfgradient(config, d) 
        kweight = grad * grad * lap
        phic = tf.multiply(kweight, dlin_k)
        phi_ii.append(tf.multiply(tf.spectral.irfft3d(phic), nc**3))

    for d in range(3):
        source = tf.add(source, tf.multiply(phi_ii[D1[d]], phi_ii[D2[d]]))
    
#     return source
    # free memory
    phi_ii = []

    # off-diag terms
    for d in range(3):
        gradi = tfgradient(config, D1[d])
        gradj = tfgradient(config, D2[d])
        kweight = gradi * gradj * lap
        phic = tf.multiply(kweight, dlin_k)
        phi = tf.multiply(tf.spectral.irfft3d(phic), nc**3)
        tf.multiply(tf.spectral.irfft3d(phic), nc**3)
        source = tf.subtract(source, tf.multiply(phi, phi))

    source = tf.multiply(source, 3.0/7.)
    return tf.multiply(tf.spectral.rfft3d(source), 1/nc**3)




def tflpt2source(dlin_k, config):
    """ Generate the second order LPT source term.  """

    bs, nc = config['boxsize'], config['nc']
    source = tf.zeros((nc, nc, nc))
    
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []

    # diagnoal terms
    lap = tflaplace(config)
    
    for d in range(3):
        grad = tfgradient(config, d) 
        kweight = grad * grad * lap
        phic = tf.multiply(kweight, dlin_k)
        phi_ii.append(tf.multiply(tf.cast(tf.spectral.ifft3d(phic), tf.float32), nc**3))


    for d in range(3):
        source = tf.add(source, tf.multiply(phi_ii[D1[d]], phi_ii[D2[d]]))
    
#     return source
    # free memory
    phi_ii = []
    # off-diag terms
    for d in range(3):
        gradi = tfgradient(config, D1[d])
        gradj = tfgradient(config, D2[d])
        kweight = gradi * gradj * lap
        phic = tf.multiply(kweight, dlin_k)
        phi = tf.multiply(tf.cast(tf.spectral.ifft3d(phic), tf.float32), nc**3)
        source = tf.subtract(source, tf.multiply(phi, phi))

    source = tf.multiply(source, 3.0/7.)
    return tf.multiply(tf.spectral.fft3d(tf.cast(source, tf.complex64)), 1/nc**3)


def tflptz0(lineark, Q, config, a=1, order=2):
    '''one step 2 LPT displacement to z=0'''
    bs, nc = config['boxsize'], config['nc']
    pos = Q*nc/bs
    
    DX1 = 1 * tflpt1(lineark, pos, config)
    if order == 2: DX2 = 1 * tflpt1(tflpt2source(lineark, config), pos, config)
    else: DX2 = 0
    return tf.add(DX1 , DX2)
    
    
###############################################################################################
# NBODY


def tflptinit(lineark, Q, a, config, order=2):
        """ This computes the 'force' from LPT as well. """
        assert order in (1, 2)

        bs, nc = config['boxsize'], config['nc']
        pos = Q * nc/bs
        dtype = np.float32

        pt = PerturbationGrowth(config['cosmology'], a=[a], a_normalize=1.0)
        DX = tf.multiply(dtype(pt.D1(a)) , tflpt1(lineark, pos, config)) 
        P = tf.multiply(dtype(a ** 2 * pt.f1(a) * pt.E(a)) , DX)
        F = tf.multiply(dtype(a ** 2 * pt.E(a) * pt.gf(a) / pt.D1(a)) , DX)
        if order == 2:
            DX2 = tf.multiply(dtype(pt.D2(a)) , tflpt1(tflpt2source(lineark, config), pos, config))
            P2 = tf.multiply(dtype(a ** 2 * pt.f2(a) * pt.E(a)) , DX2)
            F2 = tf.multiply(dtype(a ** 2 * pt.E(a) * pt.gf2(a) / pt.D2(a)) , DX2)
            DX = tf.add(DX, DX2)
            P = tf.add(P, P2)
            F = tf.add(F, F2)
            
        X = tf.add(DX, Q)
        return tf.stack((X, P, F), axis=0)




def tfKick(state, ai, ac, af, config, dtype=np.float32):
    pt = PerturbationGrowth(config['cosmology'], a=[ai, ac, af], a_normalize=1.0)
    fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
    indices = tf.constant([[1]])
    update = tf.expand_dims(tf.multiply(dtype(fac), state[2]), axis=0)
    shape = state.shape
#    print('indices = ', indices)
#    print('update = ', update)
#    print('shape = ', shape)
    update = tf.scatter_nd(indices, update, shape)
    state = tf.add(state, update)
    return state

def tfDrift(state, ai, ac, af, config, dtype=np.float32):
    pt = PerturbationGrowth(config['cosmology'], a=[ai, ac, af], a_normalize=1.0)
    fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
    indices = tf.constant([[0]])
    update = tf.expand_dims(tf.multiply(dtype(fac), state[1]), axis=0)
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    state = tf.add(state, update)
    return state


#def tfForce(state, ai, ac, af, config, dtype=np.float32):
#    var = tf.Variable(0, dtype=tf.float32)
#    var = tf.assign(var, state, validate_shape=False)
#
#    bs, nc = config['boxsize'], config['nc']
#    rho = tf.zeros((nc, nc, nc))
#    wts = np.ones(nc**3).astype(np.float32)
#    nbar = nc**3/bs**3
#
#    rho = cic_paint(rho, tf.multiply(state[0], nc/bs), wts)
#    #rho = tf.multiply(rho, 1/nbar) # I am not sure why this is not needed here
#    delta_k = tf.multiply(tf.spectral.fft3d(tf.cast(rho, tf.complex64)), 1/nc**3)
#    fac = dtype(1.5 * config['cosmology'].Om0)
#    update = tflongrange(config, tf.multiply(state[0], nc/bs), delta_k, split=0, factor=fac)
#    var = tf.scatter_update(var, 2, update)
#    return var
#
def tfForce(state, ai, ac, af, config, dtype=np.float32):

    bs, nc = config['boxsize'], config['nc']
    rho = tf.zeros((nc, nc, nc))
    wts = np.ones(nc**3).astype(np.float32)
    nbar = nc**3/bs**3

    rho = cic_paint(rho, tf.multiply(state[0], nc/bs), wts)
    ##rho = tf.multiply(rho, 1/nbar)  ###I am not sure why this is not needed here
    delta_k = tf.multiply(tf.spectral.fft3d(tf.cast(rho, tf.complex64)), 1/nc**3)
    fac = dtype(1.5 * config['cosmology'].Om0)
    update = tflongrange(config, tf.multiply(state[0], nc/bs), delta_k, split=0, factor=fac)
    

    update = tf.expand_dims(update, axis=0)

    indices = tf.constant([[2]])
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    mask = tf.stack((tf.ones_like(state[0]), tf.ones_like(state[0]), tf.zeros_like(state[0])), axis=0)
    state = tf.multiply(state, mask)
    state = tf.add(state, update)
    return state


#def tfKick(X, P, F, ai, ac, af, config, dtype=np.float32):
#    #var = tf.Variable(0, dtype=tf.float32)
#    #var = tf.assign(var, P, validate_shape=False)
#    pt = PerturbationGrowth(config['cosmology'], a=[ai, ac, af], a_normalize=1.0)
#    fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
#    update = tf.multiply(dtype(fac), F)
#    P = tf.add(P, update)
#    #var = tf.scatter_add(var, 1, update)
#    return X, P, F
#
#
#def tfDrift(X, P, F, ai, ac, af, config, dtype=np.float32):
#    #var = tf.Variable(0, dtype=tf.float32)
#    #var = tf.assign(var, state, validate_shape=False)
#    pt = PerturbationGrowth(config['cosmology'], a=[ai, ac, af], a_normalize=1.0)
#    fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
#    update = tf.multiply(dtype(fac), P)
#    #var = tf.scatter_add(var, 0, update)
#    X = tf.add(X, update)
#    return X, P, F
#
#    
#
#def tfForce(X, P, F, ai, ac, af, config, dtype=np.float32):
#    #var = tf.Variable(0, dtype=tf.float32)
#    #var = tf.assign(var, state, validate_shape=False)
#
#    bs, nc = config['boxsize'], config['nc']
#    rho = tf.zeros((nc, nc, nc))
#    wts = np.ones(nc**3).astype(np.float32)
#    nbar = nc**3/bs**3
#
#    rho = cic_paint(rho, tf.multiply(X, nc/bs), wts)
#    #rho = tf.multiply(rho, 1/nbar) # I am not sure why this is not needed here
#    delta_k = tf.multiply(tf.spectral.rfft3d(rho), 1/nc**3)
#    fac = dtype(1.5 * config['cosmology'].Om0)
#    F = tflongrange(config, tf.multiply(X, nc/bs), delta_k, split=0, factor=fac)
#    return X, P, F
#
#

def leapfrog(stages):
    """ Generate a leap frog stepping scheme.
        Parameters
        ----------
        stages : array_like
            Time (a) where force computing stage is requested.
    """
    if len(stages) == 0:
        return

    ai = stages[0]
    # first force calculation for jump starting
    yield 'F', ai, ai, ai
    x, p, f = ai, ai, ai

    for i in range(len(stages) - 1):
        a0 = stages[i]
        a1 = stages[i + 1]
        ah = (a0 * a1) ** 0.5
        yield 'K', p, f, ah
        p = ah
        yield 'D', x, p, a1
        x = a1
        yield 'F', f, x, a1
        f = a1
        yield 'K', p, f, a1
        p = a1


def tfnbody(state, config, verbose=False):
    stepping = leapfrog(config['stages'])
    actions = {'F':tfForce, 'K':tfKick, 'D':tfDrift}
    
    for action, ai, ac, af in stepping:
        if verbose: print(action, ai, ac, af)
        state = actions[action](state, ai, ac, af, config)
    return state
        #X, P, F = actions[action](X, P, F, ai, ac, af, config)
    #return X, P, F


