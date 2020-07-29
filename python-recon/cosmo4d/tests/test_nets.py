from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
from runtests.mpi import MPITest
import numpy
import logging

from cosmo4d.lab import maphd as map
from sklearn.externals import joblib

logger.setLevel(level=logging.WARNING)

from cosmo4d.pmeshengine import ParticleMesh, RealField, ComplexField, check_grad

from cosmo4d.engine import FastPMEngine
from nbodykit.cosmology import PerturbationGrowth
from nbodykit.cosmology import Planck15

cosmo = Planck15.clone(Tcmb0=0)

pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4, 4), dtype='f8')

def pk(k):
    p = ((k + 1e-9)/ 0.01) ** -3 * 80000
    return p

pt = PerturbationGrowth(cosmo)

from nbodykit.source.mesh.field import FieldMesh
from nbodykit.algorithms.fftpower import FFTPower

from cosmo4d.pmeshengine import Literal


@MPITest(commsize=[1, 4])
def test_features(comm):
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4, 4), dtype='f8', comm=comm)

    engine = FastPMEngine(pm, shift=0.5, B=1)
    code = CodeSegment(engine)
    code.find_neighbours(field='whitenoise', features='features')

    field = pm.generate_whitenoise(seed=1234, unitary=True).c2r()

    field[...] = numpy.arange(field.csize).reshape(field.cshape)[field.slices]
    eps = field.cnorm() ** 0.5 * 1e-5

    features = code.compute('features', init={'whitenoise':field})

    check_grad(code, 'features', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)




def test_apply_nets_regression():
    '''Checks only the apply_nets without prob and classify
    '''
    pm = ParticleMesh(BoxSize=32.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    #set up a network with 'nft' features, 'nl' hidden layers of size 'ls'
    #and data set of size 'nd'
    nft, nd = 5, 100
    nl, lhs = 2, [20, 30]
    ls = [nft]+ lhs + [1]
    if nl != len(lhs):
        print('Number of layers is not the same as list of layer sizes')
    wts, bias = [], []
    nx, ny = nft, ls[0]
    for i in range(nl+1):
        nx, ny = ls[i], ls[i+1]
        wt = numpy.random.uniform(size = nx*ny).reshape(nx, ny).copy()
        wts.append(wt)
        bt = numpy.random.uniform(size = ny).copy()
        bias.append(bt)
    #wts[-1] = wts[-1].reshape(-1)
    bias[-1] = bias[-1].reshape(-1)

    print(nft, nl, nd)
    for i in range(nl+1):
        print(wts[i].shape, bias[i].shape)

    acts = ['relu', 'relu']
    arch = tuple(zip(wts, bias, acts))
    features = numpy.random.uniform(size=nd*nft).reshape(nd, nft)
    features -= features.mean(axis=0)
    features /= features.std(axis=0)
    
    print(features.shape)

    #code.apply_nets(predict='predict', features='features', coeff=wts, \
    #                intercept=bias, Nd=nd, prob=False, classify=False)
    code.apply_nets(predict='predict', features='features', arch = arch, Nd=nd)

    eps = 1e-6
    check_grad(code, 'predict', 'features', init={'features': features}, eps=eps,
                rtol=1e-8, atol = 1e-8)


def test_apply_nets_prob():
    '''Checks only the apply_nets with prob and without classify
    '''
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    #set up a network with 'nft' features, 'nl' hidden layers of size 'ls'
    #and data set of size 'nd'
    nft, nd = 5, 100
    nl, lhs = 2, [20, 30]
    ls = [nft]+ lhs + [1]
    if nl != len(lhs):
        print('Number of layers is not the same as list of layer sizes')
    wts, bias = [], []
    nx, ny = nft, ls[0]
    for i in range(nl+1):
        nx, ny = ls[i], ls[i+1]
        wt = numpy.random.uniform(size = nx*ny).reshape(nx, ny).copy()
        wts.append(wt)
        bt = numpy.random.uniform(size = ny).copy()
        bias.append(bt)
    #wts[-1] = wts[-1].reshape(-1)
    bias[-1] = bias[-1].reshape(-1)

    print(nft, nl, nd)
    for i in range(nl+1):
        print(wts[i].shape, bias[i].shape)

    features = numpy.random.uniform(size=nd*nft).reshape(nd, nft)
    features -= features.mean(axis=0)
    features /= features.std(axis=0)
    
    code.apply_nets(predict='predict', features='features', coeff=wts, \
                    intercept=bias, Nd=nd, prob=True, classify=False)

    eps = 1e-4
    check_grad(code, 'predict', 'features', init={'features': features}, eps=eps,
                rtol=1e-8, atol = 1e-12)



def test_apply_nets_classification():
    '''Checks only the apply_nets with prob and with classify
    '''
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    #set up a network with 'nft' features, 'nl' hidden layers of size 'ls'
    #and data set of size 'nd'
    nft, nd = 5, 100
    nl, lhs = 2, [20, 30]
    ls = [nft]+ lhs + [1]
    if nl != len(lhs):
        print('Number of layers is not the same as list of layer sizes')
    wts, bias = [], []
    nx, ny = nft, ls[0]
    for i in range(nl+1):
        nx, ny = ls[i], ls[i+1]
        wt = numpy.random.uniform(size = nx*ny).reshape(nx, ny).copy()
        wts.append(wt)
        bt = numpy.random.uniform(size = ny).copy()
        bias.append(bt)
    #wts[-1] = wts[-1].reshape(-1)
    #bias[-1] = bias[-1].reshape(-1)

    print(nft, nl, nd)
    for i in range(nl+1):
        print(wts[i].shape, bias[i].shape)

    features = numpy.random.uniform(size=nd*nft).reshape(nd, nft)
    features -= features.mean(axis=0)
    features /= features.std(axis=0)
    
    code.apply_nets(predict='predict', features='features', coeff=wts, \
                    intercept=bias, Nd=nd, prob=True, classify=True)

    eps = 1e-4
    check_grad(code, 'predict', 'features', init={'features': features}, eps=eps,
                rtol=1e-8, atol = 1e-12)


###
###def test_halomodel():
###    bs, nc = 200., 64
###    f= 16
###    ncf = int(nc/f)
###    seed, nsteps = 100, 5
###    pm = ParticleMesh(BoxSize=bs, Nmesh=(ncf, ncf, ncf), dtype='f8')
###
###    engine = FastPMEngine(pm)
###    code = CodeSegment(engine)
###    
###
###    finalfull = map.Observable.load('../../output/example2/L%04d_N%04d_S%04d_%02dstep/'%(bs, nc, seed, nsteps) + 'datap').d
###    
###    x1, y1, z1 = int(20), int(20), int(20)
###    tmp = finalfull[...][x1:x1+ncf, y1:y1+ncf, z1:z1+ncf]
###    print('tmp -', tmp.shape)
###    final = engine.pm.generate_whitenoise(seed=123).c2r()
###    print('pm -' , pm.Nmesh)
###    print('final -' , final.shape)
###    #final.value[:] = tmp[:]
###    print(final[...].shape)
###
###    mdict = joblib.load('/global/u1/c/chmodi/Programs/cosmo4d/output/example2/L0200_N0064_S0100_05step/train/reg_nonzeromask_ftl-3.pkl')
###    pdict = joblib.load('/global/u1/c/chmodi/Programs/cosmo4d/output/example2/L0200_N0064_S0100_05step/train/cls_balanced27gridpt_ftl-3.pkl' )
###    R1, R2 = [float(pdict['smoothing'][i]) for i in range(2)]
###    pmodel = pdict['model']
###    pcoef, pintercept = pmodel.coefs_, pmodel.intercepts_
###    pmx, psx = pdict['norm']['mx'], pdict['norm']['sx']
###
###    mmodel = mdict['model']
###    mcoef, mintercept = mmodel.coefs_, mmodel.intercepts_
###    mmx, msx = mdict['norm']['mx'], mdict['norm']['sx']
###    mmy, msy = mdict['norm']['my'], mdict['norm']['sy']
###
###    posdata = [pmx, psx, pcoef, pintercept]
###    mdata = [mmx, msx, mmy, msy, mcoef, mintercept]
###
###    
###    code.apply_halomodel(model = 'model', final='final', posdata=posdata, mdata=mdata, \
###                         R1=R1, R2=R2)
###
###    print('Checking gradient now')
###    eps = 1e-4
###    check_grad(code, 'model', 'final', init={'final': final}, eps=eps,
###                rtol=1e-8, atol = 1e-10)
###
###
###
###
###def test_halomodel2():
###    bs, nc = 200., 64
###    f= 16
###    ncf = int(nc/f)
###    seed, nsteps = 100, 5
###
###    pm = ParticleMesh(BoxSize=bs/f, Nmesh=(ncf, ncf, ncf), dtype='f8')
###    engine = FastPMEngine(pm)
###    code = CodeSegment(engine)
###    
###
###    finalfull = map.Observable.load('../../output/example2/L%04d_N%04d_S%04d_%02dstep/'%(bs, nc, seed, nsteps) + 'datap').d    
###    x1, y1, z1 = int(20), int(20), int(20)
###    tmp = finalfull[...][x1:x1+ncf, y1:y1+ncf, z1:z1+ncf]
###    print('tmp -', tmp.shape)
###
###    final = engine.pm.generate_whitenoise(seed=1234).c2r()
###
###
###    print('pm -' , pm.Nmesh)
###    print('final -' , final.shape)
###    #final.value[:] = tmp[:]
###    print(final[...].shape)
###
###    mdict = joblib.load('/global/u1/c/chmodi/Programs/cosmo4d/output/example2/L0200_N0064_S0100_05step/train/reg_nonzeromask_ftl-3.pkl')
###    pdict = joblib.load('/global/u1/c/chmodi/Programs/cosmo4d/output/example2/L0200_N0064_S0100_05step/train/cls_balanced27gridpt_ftl-3.pkl' )
###    R1, R2 = [float(pdict['smoothing'][i]) for i in range(2)]
###    pmodel = pdict['model']
###    pcoef, pintercept = pmodel.coefs_, pmodel.intercepts_
###    pmx, psx = pdict['norm']['mx'], pdict['norm']['sx']
###
###    mmodel = mdict['model']
###    mcoef, mintercept = mmodel.coefs_, mmodel.intercepts_
###    mmx, msx = mdict['norm']['mx'], mdict['norm']['sx']
###    mmy, msy = mdict['norm']['my'], mdict['norm']['sy']
###
###    posdata = [pmx, psx, pcoef, pintercept]
###    mdata = [mmx, msx, mmy, msy, mcoef, mintercept]
###
###
###    pmx, psx, pcoef, pintercept = posdata
###    mmx, msx, mmy, msy, mcoef, mintercept = mdata
###
###    ##Generate differet smoothed fields

    #code.assign(x=Literal(final), y='final')
###    code.r2c(real='final', complex='d_k')
###    code.de_cic(deconvolved='decic', d_k='d_k')
###    code.r2c(real='decic', complex='d_k')
###    code.fingauss_smoothing(smoothed='R1', R=R1, d_k='d_k')
###    code.fingauss_smoothing(smoothed='R2', R=R2, d_k='d_k')
###    #code.multiply(x1='R2', x2=Literal(-1), y='negR2')
###    #code.add(x1='R1', x2='negR2', y='R12')
###
###    ##Create feature array of 27neighbor field for all 
###    names = ['final', 'R1', 'R2']
###    N = len(engine.q)
###    Nf, Nnb = len(names), 27
###    Ny = Nf*Nnb
###    code.assign(x=Literal(numpy.zeros((N, Ny))), y='pfeature')
###    code.assign(x=Literal(numpy.zeros((N, Nf))), y='mfeature')
###    grid = engine.pm.generate_uniform_particle_grid(shift=0)
###    layout = engine.pm.decompose(grid)
###
###    for i in range(Nf):
###        #p   
###        code.find_neighbours(field=names[i], features='ptmp')
###
###        #normalize feature 
###        code.add(x1='ptmp', x2=Literal(-1*pmx[i*Nnb:(i+1)*Nnb]), y='ptmp1')
###        code.multiply(x1='ptmp1', x2=Literal(psx[i*Nnb:(i+1)*Nnb]**-1), y='ptmp2')
###        code.assign_chunk(attribute='pfeature', value='ptmp2', start=i*Nnb, end=Nnb*(i+1))
###
###        #m 
###        code.readout(x=Literal(grid), mesh=names[i], value='mtmp', layout=Literal(layout), resampler='nearest')
###
###        #normalize feature
###        code.add(x1='mtmp', x2=Literal(-1*mmx[i]), y='mtmp1')
###        code.multiply(x1='mtmp1', x2=Literal(psx[i]**-1), y='mtmp2')
###        code.assign_component(attribute='mfeature', value='mtmp2', dim=i)
###
###    code.apply_nets(predict='ppredict', features='pfeature', coeff=pcoef, intercept=pintercept, Nd=N, prob=True, classify=False)
###    code.apply_nets(predict='mpredict', features='mfeature', coeff=mcoef, intercept=mintercept, Nd=N, prob=False)
###
###    #renormalize mass 
###    code.multiply(x1='mpredict', x2=Literal(msy), y='mpredict')
###    code.add(x1='mpredict', x2=Literal(mmy), y='mpredict')
###    code.reshape_scalar(x='ppredict', y='ppredict')
###    code.reshape_scalar(x='mpredict', y='mpredict')
###
###    #paint 
###    code.paint(x=Literal(grid), mesh='posmesh', layout=Literal(layout), mass='ppredict')
###    code.paint(x=Literal(grid), mesh='massmesh', layout=Literal(layout), mass='mpredict')
###    code.multiply(x1='posmesh', x2='massmesh', y='premodel')
###
###    #Smooth
###    #code.assign(x='premodel', y='model')                                                                                                                  
###    code.r2c(real='premodel', complex='d_k')
###    code.fingauss_smoothing(smoothed='model', R=4, d_k='d_k')
###
###    print('Checking gradient now')
###
###    eps = 1e-6
###    #eps = final.cnorm() ** 0.5 * 1e-3
###    check_grad(code, 'decic', 'final', init={'final': final}, eps=eps,
###                rtol=1e-8, atol =1e-6)
###
###
###

