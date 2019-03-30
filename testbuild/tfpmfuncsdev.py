# Module storing a few tensorflow function to implement FastPM
import numpy as np
import numpy
import tensorflow as tf




def r2c3d(rfield, config, dtype=tf.complex64):
    nc = config['nc']
    cfield = tf.multiply(tf.spectral.fft3d(tf.cast(rfield, dtype)), 1/config['nc']**3)
    return cfield


def c2r3d(cfield, config, dtype=tf.float32):
    nc = config['nc']
    rfield = tf.multiply(tf.cast(tf.spectral.ifft3d(cfield), dtype), config['nc']**3)
    return rfield
    

def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
    """ return k_vector given a shape (nc, nc, nc) and boxsize
    """
    k = []
    for d in range(len(shape)):
        kd = numpy.fft.fftfreq(shape[d])
        kd *= 2 * numpy.pi / boxsize * shape[d]
        kdshape = numpy.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)
        
        k.append(kd.astype(dtype))
    del kd, kdshape
    return k

def cic_paint(mesh, part, weight=None, cube_size=None, boxsize=None):
    """
        - mesh is a cube of format tf.Variable
        - part is a list of particles (:, 3), positions assumed to be in 
    mesh units if boxsize is None
        - weight is a list of weights (:)
        - cube_size is the size of the cube in mesh units
    """

    if weight is None: weight = np.ones(part.shape[0].value, dtype=part.dtype)
    if cube_size is None: cube_size = mesh.shape[0].value
    nc = int(cube_size)
    if boxsize is not None:
        part = tf.multiply(part, nc/boxsize)
    
    # Extract the indices of all the mesh points affected by each particles
    i000 = tf.cast(tf.floor(part), dtype=tf.int32)
    i100 = i000 + tf.constant([1, 0, 0])
    i010 = i000 + tf.constant([0, 1, 0])
    i001 = i000 + tf.constant([0, 0, 1])
    i110 = i000 + tf.constant([1, 1, 0])
    i101 = i000 + tf.constant([1, 0, 1])
    i011 = i000 + tf.constant([0, 1, 1])
    i111 = i000 + tf.constant([1, 1, 1])
    neighboor_coords = tf.stack([i000, i100, i010, i001,
                                 i110, i101, i011, i111], axis=1)
    kernel = 1. - tf.abs(tf.expand_dims(part, axis=1) - tf.cast(neighboor_coords, tf.float32))
    kernel = tf.reduce_prod(kernel, axis=-1, keepdims=False)
    kernel = tf.multiply(tf.expand_dims(weight, axis=1) , kernel)
        
    neighboor_coords = neighboor_coords % cube_size

    update = tf.scatter_nd(neighboor_coords, kernel, [nc, nc, nc])
    mesh = tf.add(mesh, update)
    return mesh





def cic_readout(mesh, part, cube_size=None):
    """
        - mesh is a cube
        - part is a list of particles (:, 3), positions in mesh units
        - cube_size is the size of the cube in mesh units
    """

    if cube_size is None: cube_size = mesh.shape[0].value
    # Extract the indices of all the mesh points affected by each particles
    i000 = tf.cast(tf.floor(part), dtype=tf.int32)
    i100 = i000 + tf.constant([1, 0, 0])
    i010 = i000 + tf.constant([0, 1, 0])
    i001 = i000 + tf.constant([0, 0, 1])
    i110 = i000 + tf.constant([1, 1, 0])
    i101 = i000 + tf.constant([1, 0, 1])
    i011 = i000 + tf.constant([0, 1, 1])
    i111 = i000 + tf.constant([1, 1, 1])
    neighboor_coords = tf.stack([i000, i100, i010, i001,
                                 i110, i101, i011, i111], axis=1)
    kernel = 1. - tf.abs(tf.expand_dims(part, axis=1) - tf.cast(neighboor_coords, tf.float32))
    kernel = tf.reduce_prod(kernel, axis=-1, keepdims=False)
    
#     if cube_size is not None:
    neighboor_coords = neighboor_coords % cube_size

    meshvals = tf.gather_nd(mesh, neighboor_coords)
    weightedvals = tf.multiply(meshvals, tf.cast(kernel, meshvals.dtype))
    value = tf.reduce_sum(weightedvals, axis=1)
    return value


def tflaplace(config):
    kvec = config['kvec']
    kk = sum(ki**2 for ki in kvec)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    wts = 1/kk
    imask = (~(kk==0)).astype(int)
    wts *= imask
    return wts
#     b = tf.multiply(v, 1/kk)
#     b = tf.multiply(b, imask)
#     return b 


def tfgradient(config, dir):
    kvec = config['kvec']
    bs, nc = config['boxsize'], config['nc']
    cellsize = bs/nc
    w = kvec[dir] * cellsize
    a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
    wts = a*1j
    return wts
    # a is already zero at the nyquist to ensure field is real
#     return tf.multiply(v , ac)
    


def tffknlongrange(config, r_split):
    kk = sum(ki ** 2 for ki in config['kvec'])
    if r_split != 0:
        return numpy.exp(-kk * r_split**2)
    else:
        return np.ones_like(kk)

def tflongrange(config, x, delta_k, split=0, factor=1):
    """ like long range, but x is a list of positions """
    # use the four point kernel to suppresse artificial growth of noise like terms

    ndim = 3
    lap = tflaplace(config)
    fknlrange = tffknlongrange(config, split)
    kweight = lap * fknlrange    
    pot_k = tf.multiply(delta_k, kweight)

    #var = tf.Variable(0, dtype=tf.float32)
    f = []
    for d in range(ndim):
        force_dc = tf.multiply(pot_k, tfgradient(config, d))
        #forced = tf.multiply(tf.spectral.irfft3d(force_dc), config['nc']**3)
        forced = tf.multiply(tf.cast(tf.spectral.ifft3d(force_dc), tf.float32), config['nc']**3)
        force = cic_readout(forced, x)
        f.append(force)
    
    f = tf.stack(f, axis=1)
    f = tf.multiply(f, factor)
    return f



   
#
#def tflaplace(v, config):
#    kvec = config['kvec']
#    kk = sum(ki**2 for ki in kvec)
#    mask = (kk == 0).nonzero()
#    kk[mask] = 1
#    imask = (~(kk==0)).astype(int)
#    b = tf.multiply(v, 1/kk)
#    b = tf.multiply(b, imask)
#    return b 
#
#
#def tfgradient(v, dir, config):
#    kvec = config['kvec']
#    bs, nc = config['boxsize'], config['nc']
#    cellsize = bs/nc
#    w = kvec[dir] * cellsize
#    a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
#    ac = a*1j
#    # a is already zero at the nyquist to ensure field is real
#    return tf.multiply(v , ac)
#    

##
##def cic_paint(mesh, part, weight, cube_size=None):
##    """
##        - mesh is a cube
##        - part is a list of particles (:, 3)
##        - weight is a list of weights (:)
##    """
##
##    # Create a variable to store the input mesh
##    var = tf.Variable(0, dtype=tf.float32)
##    var = tf.assign(var, mesh, validate_shape=False)
##
##    # Extract the indices of all the mesh points affected by each particles
##    i000 = tf.cast(tf.floor(part), dtype=tf.int32)
##    i100 = i000 + tf.constant([1, 0, 0])
##    i010 = i000 + tf.constant([0, 1, 0])
##    i001 = i000 + tf.constant([0, 0, 1])
##    i110 = i000 + tf.constant([1, 1, 0])
##    i101 = i000 + tf.constant([1, 0, 1])
##    i011 = i000 + tf.constant([0, 1, 1])
##    i111 = i000 + tf.constant([1, 1, 1])
##    neighboor_coords = tf.stack([i000, i100, i010, i001,
##                                 i110, i101, i011, i111], axis=1)
##
##    kernel = 1. - tf.abs(tf.expand_dims(part, axis=1) - tf.cast(neighboor_coords, tf.float32))
##    kernel = tf.reduce_prod(kernel, axis=-1, keepdims=False)
##    kernel = tf.expand_dims(weight, axis=1) * kernel
##
##    if cube_size is not None:
##        neighboor_coords = neighboor_coords % cube_size
##
##    updated_mesh = tf.scatter_nd_add(var, tf.reshape(neighboor_coords, (-1, 3)),
##                                     tf.reshape(kernel, (-1,)))
##    return updated_mesh
##


#
#def checkpaint():
#    from pmesh.pm import ParticleMesh
#    bs = 50
#    nc = 16
#    pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
#
#    nparticle = 100
#    pos = bs*np.random.random(3*nparticle).reshape(-1, 3).astype(np.float32)
#    wts = np.random.random(nparticle).astype(np.float32)
#    
#    pmmesh = pm.paint(pos, mass=wts)
#    
#    tfmesh = tf.zeros((nc, nc, nc), dtype=tf.float32)
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        tfmesh = sess.run(cic_paint(tfmesh, pos*nc/bs, weight=wts))
#    
#    print(abs(pmmesh[...] - tfmesh).sum())
#
#
#def checkreadout():
#    from pmesh.pm import ParticleMesh
#    bs = 50
#    nc = 16
#    pm = ParticleMesh(BoxSize=bs, Nmesh = [nc, nc, nc], dtype='f4')
#
#    nparticle = 100
#    pos = bs*np.random.random(3*nparticle).reshape(-1, 3).astype(np.float32)
#    base = 100*np.random.random(nc**3).reshape(nc, nc, nc).astype(np.float32)
#    
#    pmmesh = pm.create(mode='real', value=base)    
#    pmread = pmmesh.readout(pos)
#    
#    tfmesh = tf.constant(base, dtype=tf.float32)
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        tfread = sess.run(cic_readout(tfmesh, pos*nc/bs))
#    
#    print(abs((pmread[...] - tfread)/pmread).sum())
##     print(abs(pmread[...] - tfread).sum()
#


