import numpy as np
import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#from pmesh.pm import ParticleMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
#from nbodykit.lab import BigFileMesh, BigFileCatalog, FFTPower
#from nbodykit.cosmology import Planck15, EHPower, Cosmology

import sys
import tools
#import za
#import features as ft
#cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
#cosmo = Cosmology.from_dict(cosmodef)


#########################################
def diracdelta(i, j):
    if i == j: return 1
    else: return 0


def shear(base):                                                                                                                                          
    '''Takes in a PMesh object in real space. Returns am array of shear'''          
    s2 = np.zeros_like(base)
    nc = base.shape[0]
    kk = tools.fftk([nc, nc, nc], boxsize=1)
    k2 = sum(ki**2 for ki in kk)                                                                          
    k2[0,0,0] =  1
    basec = np.fft.rfftn(base, norm='ortho')
    for i in range(3):
        for j in range(i, 3):                                                       
            tmp = basec * (kk[i]*kk[j] / k2 - diracdelta(i, j)/3.)              
            baser = np.fft.irfftn(tmp, norm='ortho')
            s2[...] += baser**2                                                        
            if i != j:                                                              
                s2[...] += baser**2                                                    
                                                                                    
    return s2  


def getbias(bs, nc, hmesh, basemesh, pos, doed=False, fpos=None, kmax=0.3):

    
    print('Will fit for bias now')

    try: d0, d2, s2 = basemesh
    except:
        d0 = basemesh.copy()
        d0 -= basemesh.mean()
        d2 = 1.*d0**2
        d2 -= d2.mean()
        s2 = shear(d0)
        s2 -= 1.*d0**2
        s2 -= s2.mean()

    print(hmesh.shape)
    k, ph = tools.power(hmesh, boxsize = bs)    
    ik = numpy.where(k > kmax)[0][0]

    ed0 = tools.paintcic(pos, bs, nc, mass=d0.flatten())
    ed2 = tools.paintcic(pos, bs, nc, mass=d2.flatten())
    es2 = tools.paintcic(pos, bs, nc, mass=s2.flatten())
    if abs(ed0.mean()) < 1e-3: ed0 += 1
    if abs(ed2.mean()) < 1e-3: ed2 += 1
    if abs(es2.mean()) < 1e-3: es2 += 1
    
    ped0 = tools.power(ed0, boxsize=bs)[1]
    ped2 = tools.power(ed2, boxsize=bs)[1]
    pes2 = tools.power(es2, boxsize=bs)[1]

    pxed0d2 = tools.power(ed0, f2=ed2, boxsize=bs)[1]
    pxed0s2 = tools.power(ed0, f2=es2, boxsize=bs)[1]
    pxed2s2 = tools.power(ed2, f2=es2, boxsize=bs)[1]

    pxhed0 = tools.power(hmesh, f2=ed0, boxsize=bs)[1]
    pxhed2 = tools.power(hmesh, f2=ed2, boxsize=bs)[1]
    pxhes2 = tools.power(hmesh, f2=es2, boxsize=bs)[1]

    if doed:
        ed = tools.paintcic(pos, bs, nc, mass=np.ones(pos.shape[0]))
        ped = tools.power(ed, boxsize=bs)[1]
        pxhed = tools.power(hmesh, f2=ed, boxsize=bs)[1]
        pxedd0 = tools.power(ed, f2=ed0, boxsize=bs)[1]
        pxedd2 = tools.power(ed, f2=ed2, boxsize=bs)[1]
        pxeds2 = tools.power(ed, f2=es2, boxsize=bs)[1]

    def ftomin(bb, ii=ik, retp = False):
        b1, b2, bs = bb
        pred = b1**2 *ped0 + b2**2*ped2 + 2*b1*b2*pxed0d2 
        pred += bs**2 *pes2 + 2*b1*bs*pxed0s2 + 2*b2*bs*pxed2s2
        if doed: pred += ped + 2*b1*pxedd0 + 2*b2*pxedd2 + 2*bs*pxeds2 

        predx = 1*b1*pxhed0 + 1*b2*pxhed2
        predx += 1*bs*pxhes2
        if doed: predx += 1*pxhed

        if retp : return pred, predx
        chisq = (((ph + pred - 2*predx)[1:ii])**2).sum()**0.5.real
        return chisq.real

    print('Minimize\n')

#     b1, b2, bs2 = minimize(ftomin, [1, 1, 1], method='Nelder-Mead', options={'maxfev':10000}).x
    params =  minimize(ftomin, [1, 0, 0]).x

    b1, b2, bs2 = params

    print('\nBias fit params are : ', b1, b2, bs2)
    
    ed0 = tools.paintcic(pos, bs, nc, mass=d0.flatten())
    ed2 = tools.paintcic(pos, bs, nc, mass=d2.flatten())
    es2 = tools.paintcic(pos, bs, nc, mass=s2.flatten())
    if fpos is not None:
        ed0 = tools.paintcic(fpos, bs, nc, mass=d0.flatten())
        ed2 = tools.paintcic(fpos, bs, nc, mass=d2.flatten())
        es2 = tools.paintcic(fpos, bs, nc, mass=s2.flatten())
        mod = b1*ed0 + b2*ed2 + bs2*es2
    else:
        mod = b1*ed0 + b2*ed2 + bs2*es2
    if doed:
        ed = tools.paintcic(pos, bs, nc, mass=np.ones(pos.shape[0]))
        mod += ed
    
    return params, mod


def shear_nbkit(pm, base):                                                                                                                                          
    '''Takes in a PMesh object in real space. Returns am array of shear'''          
    s2 = pm.create(mode='real', value=0)                                                  
    kk = base.r2c().x
    k2 = sum(ki**2 for ki in kk)                                                                          
    k2[0,0,0] =  1                                                                  
    for i in range(3):
        for j in range(i, 3):                                                       
            basec = base.r2c()
            basec *= (kk[i]*kk[j] / k2 - diracdelta(i, j)/3.)              
            baser = basec.c2r()                                                                
            s2[...] += baser**2                                                        
            if i != j:                                                              
                s2[...] += baser**2                                                    
                                                                                    
    return s2  



def getbias_nbkit(pm, hmesh, basemesh, pos, grid, doed=False, fpos=None, kmax=0.3):

    if pm.comm.rank == 0: print('Will fit for bias now')

    try: d0, d2, s2 = basemesh
    except:
        d0 = basemesh.copy()
        d2 = 1.*basemesh**2
        d2 -= d2.cmean()
        s2 = shear(pm, basemesh)
        s2 -= 1.*basemesh**2
        s2 -= s2.cmean()

    ph = FFTPower(hmesh, mode='1d').power
    k, ph = ph['k'], ph['power']
    ik = numpy.where(k > kmax)[0][0]
    if pm.comm.rank == 0: print('Fit bias upto k= %.2f, index=%d'%(kmax, ik))

    glay, play = pm.decompose(grid), pm.decompose(pos)
    ed0 = pm.paint(pos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
    ed2 = pm.paint(pos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
    es2 = pm.paint(pos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)

    ped0 = FFTPower(ed0, mode='1d').power['power']
    ped2 = FFTPower(ed2, mode='1d').power['power']
    pes2 = FFTPower(es2, mode='1d').power['power']


    pxed0d2 = FFTPower(ed0, second=ed2, mode='1d').power['power']
    pxed0s2 = FFTPower(ed0, second=es2, mode='1d').power['power']
    pxed2s2 = FFTPower(ed2, second=es2, mode='1d').power['power']

    pxhed0 = FFTPower(hmesh, second=ed0, mode='1d').power['power']
    pxhed2 = FFTPower(hmesh, second=ed2, mode='1d').power['power']
    pxhes2 = FFTPower(hmesh, second=es2, mode='1d').power['power']

    if doed:
        ed = pm.paint(pos, mass=ones.readout(grid, resampler='nearest'))
        ped = FFTPower(ed, mode='1d').power['power']
        pxhed = FFTPower(hmesh, second=ed, mode='1d').power['power']
        pxedd0 = FFTPower(ed, second=ed0, mode='1d').power['power']
        pxedd2 = FFTPower(ed, second=ed2, mode='1d').power['power']
        pxeds2 = FFTPower(ed, second=es2, mode='1d').power['power']

    def ftomin(bb, ii=ik, retp = False):
        b1, b2, bs = bb
        pred = b1**2 *ped0 + b2**2*ped2 + 2*b1*b2*pxed0d2 
        pred += bs**2 *pes2 + 2*b1*bs*pxed0s2 + 2*b2*bs*pxed2s2
        if doed: pred += ped + 2*b1*pxedd0 + 2*b2*pxedd2 + 2*bs*pxeds2 

        predx = 1*b1*pxhed0 + 1*b2*pxhed2
        predx += 1*bs*pxhes2
        if doed: predx += 1*pxhed

        if retp : return pred, predx
        chisq = (((ph + pred - 2*predx)[1:ii])**2).sum()**0.5.real
        return chisq.real

    if pm.comm.rank == 0: print('Minimize\n')

#     b1, b2, bs2 = minimize(ftomin, [1, 1, 1], method='Nelder-Mead', options={'maxfev':10000}).x
    params =  minimize(ftomin, [1, 0, 0]).x

    b1, b2, bs2 = params

    if pm.comm.rank == 0: print('\nBias fit params are : ', b1, b2, bs2)
    
    if fpos is not None:
        glay, play = pm.decompose(grid), pm.decompose(fpos)
        ed0 = pm.paint(fpos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
        ed2 = pm.paint(fpos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        es2 = pm.paint(fpos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        mod = b1*ed0 + b2*ed2 + bs2*es2
    else:
        mod = b1*ed0 + b2*ed2 + bs2*es2
    if doed: mod += ed
    
    return params, mod





def eval_bfit(hmesh, mod, ofolder, noise=None, title=None, fsize=15, suff=None, save=True, fourier=False):

    pmod = FFTPower(mod, mode='1d').power
    k, pmod = pmod['k'], pmod['power']
    ph = FFTPower(hmesh, mode='1d').power['power']
    pxmodh = FFTPower(hmesh, second=mod, mode='1d').power['power']
    perr = FFTPower(hmesh -mod, mode='1d').power['power']

    if save:

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))


        ax[0].plot(k, pxmodh/(pmod*ph)**0.5)
        ax[0].set_ylabel('$r_{cc}$', fontsize=fsize)
        ax[0].set_ylim(0.5, 1.05)
        
        ax[1].plot(k,(pmod/ph)**0.5)
        ax[1].set_ylabel('$\sqrt{P_{mod}/P_{hh}}$', fontsize=fsize)

        ax[2].plot(k, perr)
        ax[2].set_yscale('log')
        ax[2].set_ylabel('$P_{\delta{mod}-\delta_h}$', fontsize=fsize)
        if noise is not None: ax[2].axhline(noise)

        if hmesh.pm.comm.rank == 0:
            for axis in ax:
                axis.set_xscale('log')
                axis.grid(which='both')
                axis.set_xlabel('$k$ (h/Mpc)', fontsize=fsize)
                axis.legend(fontsize=fsize)

            if title is not None: plt.suptitle(title, fontsize=fsize)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fname = ofolder + 'evalbfit'
            if fourier: fname +='-k'
            if suff is not None: fname = fname + '%s'%suff
            print(fname)
            fig.savefig(fname+'.png')

        plt.close()

        return k[1:], perr.real[1:]

    else:
        return k, ph, pmod, pxmodh, perr        




def getbiask(pm, hmesh, basemesh, pos, grid, fpos=None, ik=None):

    bs = pm.BoxSize[0]
    nc = pm.Nmesh[0]
    if pm.comm.rank == 0: print('Will fit for bias now')

    try: d0, d2, s2 = basemesh
    except:
        d0 = basemesh.copy()
        d2 = 1.*basemesh**2
        d2 -= d2.cmean()
        s2 = shear(pm, basemesh)
        s2 -= 1.*basemesh**2
        s2 -= s2.cmean()

    glay, play = pm.decompose(grid), pm.decompose(pos)
    ed0 = pm.paint(pos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
    ed2 = pm.paint(pos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
    es2 = pm.paint(pos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)


    dk = 2.0*numpy.pi/bs
    kmin = 2.0*numpy.pi/bs / 2.0
    kmax = 1.5*nc*numpy.pi/bs
#     dk, kmin = None, 0

    ph = FFTPower(hmesh, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power
    k, ph = ph['k'], ph['power']
    kedges = numpy.arange(k[0]-dk/2., k[-1]+dk/2., dk)
    
    #ed = pm.paint(pos, mass=ones.readout(grid, resampler='nearest'))
    ed0 = pm.paint(pos, mass=d0.readout(grid, resampler='nearest'))
    ed2 = pm.paint(pos, mass=d2.readout(grid, resampler='nearest'))
    es2 = pm.paint(pos, mass=s2.readout(grid, resampler='nearest'))

    #ped = FFTPower(ed, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    ped0 = FFTPower(ed0, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    ped2 = FFTPower(ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pes2 = FFTPower(es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']

    #pxedd0 = FFTPower(ed, second=ed0, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    #pxedd2 = FFTPower(ed, second=ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    #pxeds2 = FFTPower(ed, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']

    pxed0d2 = FFTPower(ed0, second=ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxed0s2 = FFTPower(ed0, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxed2s2 = FFTPower(ed2, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']

    #pxhed = FFTPower(hmesh, second=ed, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxhed0 = FFTPower(hmesh, second=ed0, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxhed2 = FFTPower(hmesh, second=ed2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']
    pxhes2 = FFTPower(hmesh, second=es2, mode='1d', kmin=kmin, dk=dk, kmax=kmax).power['power']




    def ftomink(bb, ii, retp = False):
        b1, b2, bs = bb
        pred = b1**2 *ped0 + b2**2*ped2 + 2*b1*b2*pxed0d2 
        pred += bs**2 *pes2 + 2*b1*bs*pxed0s2 + 2*b2*bs*pxed2s2

        predx = 1*b1*pxhed0 + 1*b2*pxhed2
        predx += 1*bs*pxhes2

        if retp : return pred, predx
        chisq = (((ph + pred - 2*predx)[ii])**2).real
        return chisq

    if pm.comm.rank == 0: print('Minimize\n')

    b1k, b2k, bsk = numpy.zeros_like(k), numpy.zeros_like(k), numpy.zeros_like(k)
    for ii in range(k.size):
        tfunc = lambda p: ftomink(p,ii)
        b1k[ii], b2k[ii], bsk[ii] = minimize(tfunc, [1, 0, 0]).x

    paramsk = [b1k, b2k, bsk]

    def transfer(mesh, tk):
        meshc = mesh.r2c()
        kk = meshc.x
        kmesh = sum([i ** 2 for i in kk])**0.5
#         _, kedges = numpy.histogram(kmesh.flatten(), nc)
        kind = numpy.digitize(kmesh, kedges, right=False)
        toret = mesh.pm.create(mode='complex', value=0)

        for i in range(kedges.size):
            mask = kind == i
            toret[mask] = meshc[mask]*tk[i]
        return toret.c2r()

    
    if fpos is not None:
        glay, play = pm.decompose(grid), pm.decompose(fpos)
        ed0 = pm.paint(fpos, mass=d0.readout(grid, layout = glay, resampler='nearest'), layout=play)
        ed2 = pm.paint(fpos, mass=d2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        es2 = pm.paint(fpos, mass=s2.readout(grid, layout = glay, resampler='nearest'), layout=play)
        mod = transfer(ed0, b1k) + transfer(ed2, b2k) + transfer(es2, bsk)        
    else:
        mod = transfer(ed0, b1k) + transfer(ed2, b2k) + transfer(es2, bsk)
    
    return [k, paramsk], mod

         


#$#if __name__=="__main__":
#$#
#$#    #bs, nc = 256, 128
#$#    bs, nc = 400, 512
#$#    nsteps = 5
#$#    ncf = 512
#$#    numd = 1e-3
#$#    num = int(numd*bs**3)
#$#    dpath = '/global/cscratch1/sd/chmodi/cosmo4d/data/z00/L%04d_N%04d_S0100_%02dstep/'%(bs, nc, nsteps)
#$#    hpath = '/global/cscratch1/sd/chmodi/cosmo4d/data/z00/L%04d_N%04d_S0100_40step/'%(bs, ncf)
#$#    
#$#    pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
#$#    rank = pm.comm.rank
#$#    aa = 1.0000
#$#    zz = 1/aa-1
#$#    Rsm = 0
#$#    ik = 50
#$#    ii = 50
#$#    
#$#    for masswt in [True, False]:
#$#        for zadisp in [True, False]:
#$#            suff = '%04d_%04d_n%04d'%(bs, nc, numd*1e4)
#$#    
#$#
#$#            lin = BigFileMesh(dpath + '/mesh', 's').paint()
#$#            dyn = BigFileCatalog(dpath + '/dynamic/1')
#$#            hcat = BigFileCatalog(hpath + '/FOF/')
#$#            #
#$#            hpos = hcat['PeakPosition'][1:num]
#$#            #print('Mass : ', rank, hcat['Mass'].compute())[1:num]
#$#            hmass = hcat['Mass'].compute()[1:num]
#$#            if not masswt:
#$#                hmass = hmass*0 + 1.
#$#                suff = suff + '-pos'
#$#            else: 
#$#                suff = suff + '-mass'
#$#
#$#            hlay = pm.decompose(hpos)
#$#            hmesh = pm.paint(hpos, mass=hmass, layout=hlay)
#$#            hmesh /= hmesh.cmean()
#$#
#$#            #
#$#            grid = dyn['InitPosition'].compute()
#$#            fpos = dyn['Position'].compute()
#$#            print(rank, (grid-fpos).std(axis=0))
#$#
#$#            #
#$#            dgrow = cosmo.scale_independent_growth_factor(zz)
#$#            if zadisp :
#$#                fpos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
#$#                suff = suff + '-za'
#$#            dlay = pm.decompose(fpos)
#$#
#$#
#$#            paramsza, mod = getbias(pm, basemesh=lin, hmesh=hmesh, pos=fpos, grid=grid, ik=ik)
#$#            eval_bfit(hmesh, mod, ofolder='./figs/', suff=suff)
#$#            paramsza, mod = getbiask(pm, basemesh=lin, hmesh=hmesh, pos=fpos, grid=grid, ik=ik)
#$#            eval_bfit(hmesh, mod, ofolder='./figs/', suff=suff, fourier=True)
#$#            
#$#


##
##    lin = BigFileMesh(dpath + '/mesh', 's').paint()
##    dyn = BigFileCatalog(dpath + '/dynamic/1')
##    hcat = BigFileCatalog(hpath + '/FOF/')
##    hpos = hcat['PeakPosition'][1:num]
##    hlay = pm.decompose(hpos)
##    #hmass = hcat['Mass'].compute()[1:num]
##
##    for nc2 in [128, 256, 512]:
##        for R in [20]:
##            for zadisp in [True, False]:
##
##                
##                suff = '%04d_%04d_n%04d-H%d-R%d'%(bs, nc, numd*1e4, nc2, R)
##                print(suff)
##                pm2 = ParticleMesh(BoxSize=bs, Nmesh=[nc2, nc2, nc2])
##
##                hlay2 = pm2.decompose(hpos)
##                hmesh = pm2.paint(hpos, layout=hlay2)
##                hmesh /= hmesh.cmean()
##                hmesh = ft.smooth(hmesh, R, 'gauss')
##
##                #
##                hmass = hmesh.readout(hpos, layout=hlay2)
##                del hmesh
##
##                hmesh = pm.paint(hpos, mass=hmass, layout=hlay)
##                hmesh /= hmesh.cmean()
##
##                #
##                grid = dyn['InitPosition'].compute()
##                fpos = dyn['Position'].compute()
##
##                #
##                dgrow = cosmo.scale_independent_growth_factor(zz)
##                if zadisp :
##                    fpos = za.doza(lin.r2c(), grid, z=zz, dgrow=dgrow)
##                    suff = suff + '-za'
##                dlay = pm.decompose(fpos)
##
##
##                paramsza, mod = getbias(pm, basemesh=lin, hmesh=hmesh, pos=fpos, grid=grid, ik=ik)
##
##                eval_bfit(hmesh, mod, ofolder='./figs/denswt/', suff=suff)
##
