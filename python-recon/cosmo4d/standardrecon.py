import numpy as np
from nbodykit.algorithms import FFTPower
import sys
sys.path.append('/global/homes/c/chmodi/Programs/cosmo4d/train_nets')
import features as ft
#import datalib as lib
#import datatools as dtools
#import nettools as ntools
#from nettools import *
#import sigtools as stools
#import diagnostic as dg
#

def calc_disp(i, mesh, b, pm):
    dk = mesh.r2c()
    k2 = 0
    for ki in dk.x: k2 =  k2 + ki ** 2
    k2[0, 0, 0] = 1
    sk = -(0+1j)*dk*dk.x[i]/k2
    sr = sk.c2r()
    return sr/b


def displace(pm, displist, pos, rsd=False, f=None, beta=None, mass=None):
    dispxmesh, dispymesh, dispzmesh = displist
    dispx = dispxmesh.readout(pos)
    dispy = dispymesh.readout(pos)
    dispz = dispzmesh.readout(pos)
    if rsd:
        dispz = dispz + (f -  beta)/(1 + beta)*dispz
    disp = np.array([dispx, dispy, dispz]).T
    layout = pm.decompose(pos+disp)
    if mass is not None: shiftmesh = pm.paint(pos + disp, mass=mass, layout=layout)
    else: shiftmesh = pm.paint(pos + disp, layout=layout)
    shiftmesh /= shiftmesh.cmean()
    shiftmesh -= 1
    return shiftmesh

def calc_displist(pm, base, b=1):   
    dispxmesh = calc_disp(0, base, b=b, pm=pm)
    dispymesh = calc_disp(1, base, b=b, pm=pm)
    dispzmesh = calc_disp(2, base, b=b, pm=pm)
    return [dispxmesh, dispymesh, dispzmesh]

#def calc_disp2(i, mesh, b, pm=pm, rsd=False, f=ff, beta=None):
#    dk = mesh.r2c()
#    k2 = 0
#    for ki in pm.k: k2 =  k2 + ki ** 2
#    k2[0, 0, 0] = 1
#    sk = -(0+1j)*dk*pm.k[i]/k2
#    sr = sk.c2r()
#    if rsd and i==2: sr /= (1+beta)
#    return sr/b
#
#def displace2(pm, displist, pos, rsd=False, f=ff, beta=None):
#    dispxmesh, dispymesh, dispzmesh = displist
#    dispx = dispxmesh.readout(pos)
#    dispy = dispymesh.readout(pos)
#    dispz = dispzmesh.readout(pos)
#    if rsd: dispz = dispz + f *dispz
#    disp = np.array([dispx, dispy, dispz]).T
#    shiftmesh = pm.paint(pos + disp)
#    shiftmesh /= shiftmesh.cmean()
#    shiftmesh -= 1
#    return shiftmesh
#
#def calc_displist2(base, b=1, rsd=False, f=ff, beta=None):
#    dispxmesh = calc_disp2(0, base, b=bias, rsd=rsd, f=ff, beta=beta)
#    dispymesh = calc_disp2(1, base, b=bias, rsd=rsd, f=ff, beta=beta)
#    dispzmesh = calc_disp2(2, base, b=bias, rsd=rsd, f=ff, beta=beta)
#    return [dispxmesh, dispymesh, dispzmesh]
#    
    

    

def standard(pm, fofcat, datap, mf, kb=6, Rsm = 7, rsd = False, zz=0, M= 0.3175, mass=False, poskey='PeakPosition'):

    if rsd:
        if pm.comm.rank == 0: print('\n RSD! Key used to get position is --- RSPeakPosition \n\n')            
        position = fofcat['RS%s'%poskey].compute()
    else:
        position = fofcat['%s'%poskey].compute()
    try: hmass =  fofcat['AMass'].compute()*1e10
    except: hmass = fofcat['Mass'].compute()*1e10

    pks = FFTPower(datap.s, mode='1d').power['power']
    pkf = FFTPower(datap.d, mode='1d').power['power']
    
    random = pm.generate_uniform_particle_grid()
    # random = np.random.uniform(0, 400, 3*128**3).reshape(-1, 3)
    Rbao = Rsm/2**0.5
    aa = mf.cosmo.ztoa(zz)
    ff = mf.cosmo.Fomega1(mf.cosmo.ztoa(zz))

    
    layout = pm.decompose(position)
    if mass: hmesh = pm.paint(position, mass = hmass, layout=layout)
    else: hmesh = pm.paint(position, layout=layout)
    hmesh /= hmesh.cmean()
    hmesh -= 1
    hmeshsm =  ft.smooth(hmesh, Rbao, 'gauss')

    #bias
    layout = pm.decompose(fofcat['%s'%poskey])
    hrealp = pm.paint(fofcat['%s'%poskey], layout=layout)
    hrealp /= hrealp.cmean()
    hrealp -= 1

    pkhp = FFTPower(hrealp, mode='1d').power['power']
    bias = ((pkhp[1:kb]/pkf[1:kb]).mean()**0.5).real
    beta = bias/ff
    print('bias = ', bias)

    displist = calc_displist(pm=pm, base=hmeshsm, b=bias)

    if mass: hpshift = displace(pm, displist, position, rsd=rsd, f=ff, beta=beta, mass = hmass)
    else: hpshift = displace(pm, displist, position, rsd=rsd, f=ff, beta=beta, mass = None)

    rshift = displace(pm, displist, random, rsd=rsd, f=ff, beta=beta)
    recon = hpshift - rshift
    
    return recon, hpshift, rshift




def dostd(hdict, numd, pkf, datas, kb=6, Rsm = 7, rsd = False, zz=0, M= 0.3175, mass=False, retfield=False, 
          retpower=False, mode='1d', Nmu=5, los=[0, 0, 1], retall=False):
    #propogator is divided by bias

    hpos = hdict['position']
    hmass = hdict['mass']
    pks = FFTPower(datas, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
    
    aa = mf.cosmo.ztoa(zz)
    ff = mf.cosmo.Fomega1(mf.cosmo.ztoa(zz))
    rsdfac = 100/(aa**2 * mf.cosmo.Ha(z=zz)**1)
    print('rsdfac = ', rsdfac)
    hposrsd = hdict['position'] + np.array(los)*hdict['velocity']*rsdfac
    
    layout = pm.decompose(hpos[:int(numd*bs**3)])
    if mass: hpmesh = pm.paint(hpos[:int(numd*bs**3)], mass = hmass[:int(numd*bs**3)], layout=layout)
    else: hpmesh = pm.paint(hpos[:int(numd*bs**3)], layout=layout)
    hpmesh /= hpmesh.cmean()
    hpmesh -= 1

    layout = pm.decompose(hposrsd[:int(numd*bs**3)])
    if mass: hpmeshrsd = pm.paint(hposrsd[:int(numd*bs**3)], mass = hmass[:int(numd*bs**3)], layout=layout)
    else: hpmeshrsd = pm.paint(hposrsd[:int(numd*bs**3)], layout=layout)
    hpmeshrsd /= hpmeshrsd.cmean()
    hpmeshrsd -= 1

    pkhp = FFTPower(hpmesh, mode='1d').power['power']
    bias = ((pkhp[1:kb]/pkf[1:kb]).mean()**0.5).real
    beta = bias/ff
    print('bias = ', bias)

    random = pm.generate_uniform_particle_grid()
    random = np.random.uniform(0, 400, 3*128**3).reshape(-1, 3)
    Rbao = Rsm/2**0.5

    if not rsd:
        hpmeshsm = ft.smooth(hpmesh, Rbao, 'gauss')
        displist = calc_displist(hpmeshsm, b=bias)
        if mass: 
            hpshift = displace(pm, displist, hpos[:int(numd*bs**3)], mass = hmass[:int(numd*bs**3)])
        else: 
            hpshift = displace(pm, displist, hpos[:int(numd*bs**3)])
        rshift = displace(pm, displist, random)
        recon = hpshift - rshift
        pksstd = FFTPower(recon, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
        pkxsstd = FFTPower(recon, second=datas, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
        rccstd = pkxsstd / (pks*pksstd)**0.5
        cksstd = pkxsstd / pks /bias
        
    
    if rsd:
        RSD
        hpmeshrsdsm = ft.smooth(hpmeshrsd, Rbao, 'gauss')
        displist = calc_displist(hpmeshrsdsm, b=bias)
        hpshift = displace(pm, displist, hposrsd[:int(numd*bs**3)], rsd=True, f=ff, beta=bias/ff)
        if mass: hpshift = displace(pm, displist, hposrsd[:int(numd*bs**3)], rsd=True, f=ff, beta=bias/ff, 
                                    mass = hmass[:int(numd*bs**3)])
        else: hpshift = displace(pm, displist, hposrsd[:int(numd*bs**3)], rsd=True, f=ff, beta=bias/ff)
        rshift = displace(pm, displist, random, rsd=True, f=ff, beta=beta)
        recon = hpshift - rshift
        pksstd = FFTPower(recon, mode='1d').power['power']
        pkxsstd = FFTPower(recon, second=datap[tkey].s, mode='1d').power['power']
        pksstd = FFTPower(recon, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
        pkxsstd = FFTPower(recon, second=datas, mode=mode, Nmu=Nmu, los=[0, 0, 1]).power['power']
        rccstd = pkxsstd / (pks*pksstd)**0.5
        cksstd = pkxsstd / pks / bias
        
    if retall: return [rccstd, cksstd], [recon, hpshift, rshift, displist],  [pkxsstd, pksstd, pks], bias
    if retfield: return [rccstd, cksstd], [recon, hpshift, rshift, displist]
    elif retpower: return [rccstd, cksstd], [pkxsstd, pksstd, pks], bias
    else: return rccstd, cksstd

    
    

