import numpy as np

import flowpm.utils as utils
import flowpm.tfpm as tfpm
import flowpm.kernels as kernels

import tools

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()




def standardrecon(bs, nc, base, pos, bias, R=8):

    base = base.astype(np.float32)
    pos = pos.astype(base.dtype)
    kvsym = tools.fftk([nc, nc, nc], bs, symmetric=True, dtype=base.dtype)
    basesm = tools.gauss(base, kvsym, R)
    grid = bs/nc*np.indices((nc, nc, nc)).reshape(3, -1).T.astype(base.dtype)
    grid = np.expand_dims(grid, 0)

    grid = grid *nc/bs
    pos = pos *nc/bs
    
    kv = kernels.fftk([nc, nc, nc], symmetric=False, dtype=base.dtype)
    
    mesh = tf.constant(basesm.astype(base.dtype))
    meshk = utils.r2c3d(mesh, norm=nc**3)
    
    DX = tfpm.lpt1(meshk, pos, kvec=kv)
    DX = tf.multiply(DX, -1/bias)
    pos = tf.add(pos, DX)
    displaced = tf.zeros_like(mesh)
    displaced = utils.cic_paint(displaced, pos, name='displaced')
    
    DXrandom = tfpm.lpt1(meshk, grid, kv)
    DXrandom = tf.multiply(DXrandom, -1/bias)
    posrandom = tf.add(grid, DXrandom)
    random = tf.zeros_like(mesh)
    random = utils.cic_paint(random, posrandom, name='random')
    return displaced, random





def standardinit(bs, nc, base, pos, final, R=8):

    ##
    print('Initial condition from standard reconstruction')
    
    if abs(base.mean()) > 1e-6: 
        base = (base - base.mean())/base.mean()
    pfin = tools.power(final, boxsize=bs)[1]
    ph = tools.power(1+base, boxsize=bs)[1]
    bias = ((ph[1:5]/pfin[1:5])**0.5).mean()
    print('Bias = ', bias)

    tfdisplaced, tfrandom = standardrecon(bs, nc, np.expand_dims(base, 0), np.expand_dims(pos, 0), bias, R=R)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        displaced, random = sess.run([tfdisplaced, tfrandom])

    displaced /= displaced.mean()
    displaced -= 1
    random /= random.mean()
    random -= 1
    recon = displaced - random
    return recon



if __name__=="__main__":

    bs, nc, step = 400, 128, 5
    ncf, stepf = 512, 40
    seed = 100
    numd = 1e-3
    num = int(numd*bs**3)
        
    final = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/d/')
    ic = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0128_S0100_05step/mesh/s/')

    pypath = '/global/cscratch1/sd/chmodi/cosmo4d/output/version2/L0400_N0128_05step-fof/lhd_S0100/n10/opt_s999_iM12-sm3v25off/meshes/'
    fin = tools.readbigfile(pypath + 'decic//') 
    
    hpos = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/PeakPosition//')[1:int(bs**3 *numd)]
    hmass = tools.readbigfile('/project/projectdirs/m3058/chmodi/cosmo4d/data/L0400_N0512_S0100_40step/FOF/Mass//')[1:int(bs**3 *numd)].flatten()

    meshpos = tools.paintcic(hpos, bs, nc)
    meshmass = tools.paintcic(hpos, bs, nc, hmass.flatten()*1e10)
    data = meshmass
    kv = tools.fftk([nc, nc, nc], bs, symmetric=True, dtype=np.float32)


    
    base = meshpos
    base = (base - base.mean())/base.mean()
    pfin = tools.power(final, boxsize=bs)[1]
    ph = tools.power(1+base, boxsize=bs)[1]
    bias = ((ph[1:5]/pfin[1:5])**0.5).mean()
    print(bias)

    tf.reset_default_graph()
    tfdisplaced, tfrandom = standardrecon(bs, nc, np.expand_dims(base, 0), np.expand_dims(hpos, 0), bias, R=8)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        displaced, random, dx = sess.run([tfdisplaced, tfrandom, tfdx])
        dx = dx[0]

    displaced /= displaced.mean()
    displaced -= 1
    random /= random.mean()
    random -= 1
    recon = np.squeeze(displaced - random)
            
    print(displaced.shape, random.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize = (9, 4))
    plt.subplot(131)
    plt.imshow(ic.sum(axis=0))
    plt.subplot(132)
    plt.imshow(fin.sum(axis=0))
    plt.subplot(133)
    plt.imshow(recon.sum(axis=0))
    plt.savefig('tmp.png')
    plt.close()

    print(ic.mean(),  recon.mean())
    k, p1 = tools.power(ic+1, boxsize=bs)
    p2 = tools.power(recon+1, boxsize=bs)[1]
    px = tools.power(ic+1, f2=recon+1, boxsize=bs)[1]
    plt.plot(k, p2/p1)
    plt.plot(k, px/(p1*p2)**0.5, '--')
    plt.semilogx()
    plt.savefig('tmp2.png')
