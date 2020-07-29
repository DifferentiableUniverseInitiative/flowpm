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

def test_force():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_lpt(pt=pt, dlinear_k='dlinear_k', aend=0.1, s='s', v='v', s1='s1', s2='s2')

    field = engine.pm.generate_whitenoise(seed=1234).c2r()
    s = code.compute('s', init={'whitenoise' : field})
    code = CodeSegment(engine)
    code.force(s='s', force='force', force_factor=1.0)

    eps = (pm.comm.allreduce((s ** 2).sum()) / pm.comm.allreduce(len(s))) ** 0.5 * 1e-3
    s = s.clip(2 *eps * pm.BoxSize / pm.Nmesh, (1 - 2 * eps) * pm.BoxSize / pm.Nmesh)

    check_grad(code, 'force', 's', init={'s': s}, eps=eps,
                rtol=1e-8)

#    from fastpm.operators import gravity
#    f_truth = gravity(engine.get_x(s), engine.pm, 1.0)
#    assert_allclose(force, f_truth, atol=1e-8, rtol=1e-4)


def test_create_linear_field():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.c2r(complex='dlinear_k', real='dlinear')

    field = engine.pm.generate_whitenoise(seed=1234).c2r()

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 'dlinear', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)


def test_linear_vs_whitenoise():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    def pk(k):
        return 0.0 * k + 1.0
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.create_whitenoise(whitenoise='whitenoise2', powerspectrum=pk, dlinear_k='dlinear_k')

    def tf1(k):
        k2 = sum(ki**2 for ki in k)
        r = (pk(k2 ** 0.5) / engine.pm.BoxSize.prod()) ** 0.5
        r[k2 == 0] = 1.0
        return r
    def tf2(k):
        k2 = sum(ki**2 for ki in k)
        r = (pk(k2 ** 0.5) / engine.pm.BoxSize.prod()) ** -0.5
        r[k2 == 0] = 1.0
        return r

    field = engine.pm.generate_whitenoise(seed=1234).c2r()
    eps = field.cnorm() ** 0.5 * 1e-3

    dlineark, whitenoise2 = code.compute(['dlinear_k', 'whitenoise2'], init={'whitenoise': field})
    dlineark2 = field.r2c().apply(lambda k, v: tf1(k) * v)
    assert_allclose(dlineark, dlineark2)
    assert_allclose(field, whitenoise2)

def test_solve_linear_displacement():
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)

    code.r2c(real='source', complex='dlinear_k')
    code.solve_linear_displacement(source_k='dlinear_k', s='s')

    field = pm.generate_whitenoise(seed=1234, mode='real')

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 's', 'source', init={'source': field}, eps=eps,
                rtol=1e-8)

#    from fastpm.operators import lpt1, lpt2source
    dlin_k, s = code.compute(['dlinear_k', 's'], init={'source' : field})

#    s1_truth = lpt1(dlin_k, engine.q, resampler='cic')
#    assert_allclose(s, s1_truth, rtol=1e-5)

def test_solve_lpt():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4, 4), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_lpt(pt=pt, dlinear_k='dlinear_k', aend=0.1, s='s', v='v', s1='s1', s2='s2')

    field = pm.generate_whitenoise(seed=1234).c2r()
    s1, s2 = code.compute(['s1', 's2'], init={'whitenoise' : field})
    dlin_k = code.compute('dlinear_k', init={'whitenoise' : field})

    s1, tape = code.compute('s1', init={'whitenoise' : field}, return_tape=True)

#    from fastpm.operators import lpt1, lpt2source
#    s1_truth = lpt1(dlin_k, engine.q, resampler='cic')
#    dlin2_k = lpt2source(dlin_k)
#    s2_truth = lpt1(dlin2_k, engine.q, resampler='cic')

#    assert_allclose(s1, s1_truth, rtol=1e-4)
#    assert_allclose(s2, s2_truth, rtol=1e-4)

    eps = field.cnorm() ** 0.5 * 1e-4
    check_grad(code, 's1', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)

    check_grad(code, 's2', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)

    check_grad(code, 's', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)

    check_grad(code, 'v', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)

def test_solve_fastpm_linear_growth():
    pm = ParticleMesh(BoxSize=1024.0, Nmesh=(128, 128, 128), dtype='f8')

    engine = FastPMEngine(pm)

    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_fastpm(pt=pt, dlinear_k='dlinear_k', asteps=[0.1, 0.5, 1.0], s='s', v='v', s1='s1', s2='s2')
    code.get_x(s='s', x='x')
    code.paint_simple(x='x', density='density')
    field = pm.generate_whitenoise(seed=1234, unitary=True).c2r()

    density, dlinear_k, s = code.compute(['density', 'dlinear_k', 's'], init={'whitenoise' : field})
    density_k = density.r2c()
    p_lin= FFTPower(FieldMesh(dlinear_k), mode='1d')
    p_nonlin = FFTPower(FieldMesh(density), mode='1d')

    # the simulation shall do a linear growth
    t1 = abs((p_nonlin.power['power'] / p_lin.power['power']) ** 0.5)
    assert_allclose(t1[1:4], 1.0, rtol=5e-2)

def test_generate_2nd_order_source():
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='source', complex='source_k')
    code.generate_2nd_order_source(source_k='source_k', source2_k='source2_k')
    code.c2r(complex='source2_k', real='source2')
    field = pm.generate_whitenoise(seed=1234).c2r()

#    from fastpm.operators import lpt1, lpt2source
#    dlin2_k = lpt2source(field.r2c())
    source2_k = code.compute('source2', init={'source' : field}).r2c()

#    assert_allclose(dlin2_k[...], source2_k[...], atol=1e-7, rtol=1e-4)

    check_grad(code, 'source2', 'source', init={'source': field}, eps=1e-4,
                rtol=1e-8)

def test_solve_fastpm():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4, 4), dtype='f8')

    engine = FastPMEngine(pm, shift=0.5, B=1)

    code = CodeSegment(engine)
    code.create_linear_field(whitenoise='whitenoise', powerspectrum=pk, dlinear_k='dlinear_k')
    code.solve_fastpm(pt=pt, dlinear_k='dlinear_k', asteps=[0.1, 1.0], s='s', v='v', s1='s1', s2='s2')
#    code.solve_fastpm(pt=pt, dlinear_k='dlinear_k', asteps=[1.0], s='s', v='v', s1='s1', s2='s2')
    code.get_x(s='s', x='x')
    code.paint_simple(x='x', density='density')
    field = pm.generate_whitenoise(seed=1234, unitary=True).c2r()

    eps = field.cnorm() ** 0.5 * 1e-5

    check_grad(code, 's1', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)

    check_grad(code, 's', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)
    check_grad(code, 'v', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)
    check_grad(code, 'density', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)


def test_project():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(4, 4, 4), dtype='f8')

    engine = FastPMEngine(pm, shift=0.5, B=1)

    code = CodeSegment(engine)
    code.project(field='whitenoise', projection='projection')

    field = pm.generate_whitenoise(seed=1234, unitary=True).c2r()

    eps = field.cnorm() ** 0.5 * 1e-5

    check_grad(code, 'projection', 'whitenoise', init={'whitenoise': field}, eps=eps,
                rtol=1e-8)


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


def test_reshape_scalar():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)

    x = numpy.random.uniform(100).reshape(-1, 1)
    code.reshape_scalar(x='x', y='y')

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps,
                rtol=1e-8)


def test_gauss_smoothing():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(complex='d_k', real='d_r')
    code.gauss_smoothing(smoothed='gauss', R=2, d_k='d_k')

    field = engine.pm.generate_whitenoise(seed=1234).c2r()

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 'gauss', 'd_r', init={'d_r': field}, eps=eps,
                rtol=1e-8)


def test_fingauss_smoothing():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(complex='d_k', real='d_r')
    code.fingauss_smoothing(smoothed='fingauss', R=2, d_k='d_k')

    field = engine.pm.generate_whitenoise(seed=1234).c2r()

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 'fingauss', 'd_r', init={'d_r': field}, eps=eps,
                rtol=1e-8)


def test_decic():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    code.r2c(complex='d_k', real='d_r')
    code.de_cic(deconvolved='decic', d_k='d_k')

    field = engine.pm.generate_whitenoise(seed=1234).c2r()

    eps = field.cnorm() ** 0.5 * 1e-3
    check_grad(code, 'decic', 'd_r', init={'d_r': field}, eps=eps,
                rtol=1e-8)



def test_relu():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    x = numpy.random.uniform(-1, 1, 1000)
    #x = numpy.random.uniform(2, 3, 1000)
    code.relu(y='y', x='x')

    eps = 1e-8
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps, rtol=1e-8, atol = 1e-8)

def test_elu():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    x = numpy.random.uniform(-5, 5, 1000)
    #x = numpy.random.uniform(2, 3, 1000)
    code.elu(y='y', x='x', alpha=1.3)

    eps = 1e-8
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps, rtol=1e-8, atol = 1e-8)


def test_logistic():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    x = numpy.random.uniform(-1, 1, 1000) * 5
    code.logistic(y='y', x='x', t = 2, w = 3)

    eps = 1e-8
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps, rtol=1e-8, atol = 1e-8)



def test_identity():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    x = numpy.random.uniform(-1, 1, 1000)
    code.identity(y='y', x='x')

    eps = 1e-4
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps,
                rtol=1e-8)


def test_matrix_cmul():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    nx, ny, nd = 19, 31, 100
    wt = numpy.random.uniform(size = nx*ny).reshape(nx, ny).copy()*10
    vec = numpy.random.uniform(size = nd*nx).reshape(nd, nx).copy()*10

    code.matrix_cmul(W=wt, x='x', y='y')
    eps = 1e-4
    check_grad(code, 'y', 'x', init={'x': vec}, eps=eps,
                rtol=1e-8)


def test_pow():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    x = numpy.random.uniform(-1, 1, 1000)
    #x = numpy.random.uniform(2, 3, 1000)
    #print('For power = 2')
    #code.pow(y='y', x='x', power=2)
    #eps = 1e-8
    #check_grad(code, 'y', 'x', init={'x': x}, eps=eps, rtol=1e-8, atol = 1e-8)

    print('For power = 0.5')
    x = numpy.random.uniform(0, 1, 1000)
    code.pow(y='y', x='x', power=0.5)
    eps = 1e-8
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps, rtol=1e-8, atol = 1e-8)



def test_expon():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    x = numpy.random.uniform(-4, 4, 1000)
    code.expon(y='y', x='x')
    eps = 1e-8
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps, rtol=1e-8, atol = 1e-8)



def test_log():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    
    x = numpy.random.uniform(0.001, 5, 1000)
    code.log(y='y', x='x')
    eps = 1e-8
    check_grad(code, 'y', 'x', init={'x': x}, eps=eps, rtol=1e-8, atol = 1e-8)


def test_divide():
    pm = ParticleMesh(BoxSize=8.0, Nmesh=(8, 8, 8), dtype='f8')
    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    

    x1 = numpy.random.uniform(1, 10, 1000)
    x2 = numpy.random.uniform(1, 10, 1000)
    code.divide(y='y', x1='x1', x2='x2')
    eps = 1e-8
    check_grad(code, 'y', 'x1', init={'x1': x1, 'x2': x2}, eps=eps, rtol=1e-8, atol = 1e-8)
    check_grad(code, 'y', 'x2', init={'x1': x1, 'x2': x2}, eps=eps, rtol=1e-8, atol = 1e-8)


def test_apply_nets_regression():
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
    
    print(features.shape)

    code.apply_nets(predict='predict', features='features', coeff=wts, \
                    intercept=bias, Nd=nd, prob=False, classify=False)

    eps = 1e-4
    check_grad(code, 'predict', 'features', init={'features': features}, eps=eps,
                rtol=1e-8, atol = 1e-12)


def test_apply_nets_prob():
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




def test_net_combination():
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



def test_halomodel():
    bs, nc = 200., 64
    f= 16
    ncf = int(nc/f)
    seed, nsteps = 100, 5
    pm = ParticleMesh(BoxSize=bs, Nmesh=(ncf, ncf, ncf), dtype='f4')

    engine = FastPMEngine(pm)
    code = CodeSegment(engine)
    

    finalfull = map.Observable.load('../../output/example2/L%04d_N%04d_S%04d_%02dstep/'%(bs, nc, seed, nsteps) + 'datap').d
    
    x1, y1, z1 = int(20), int(20), int(20)
    tmp = finalfull[...][x1:x1+ncf, y1:y1+ncf, z1:z1+ncf]
    print('tmp -', tmp.shape)
    final = engine.pm.generate_whitenoise(seed=1234).c2r()
    print('pm -' , pm.Nmesh)
    print('final -' , final.shape)
    #final.value[:] = tmp[:]
    print(final[...].shape)

    mdict = joblib.load('/global/u1/c/chmodi/Programs/cosmo4d/output/example2/L0200_N0064_S0100_05step/train/reg_nonzeromask_ftl-3.pkl')
    pdict = joblib.load('/global/u1/c/chmodi/Programs/cosmo4d/output/example2/L0200_N0064_S0100_05step/train/cls_balanced27gridpt_ftl-3.pkl' )
    R1, R2 = [float(pdict['smoothing'][i]) for i in range(2)]
    pmodel = pdict['model']
    pcoef, pintercept = pmodel.coefs_, pmodel.intercepts_
    pmx, psx = pdict['norm']['mx'], pdict['norm']['sx']

    mmodel = mdict['model']
    mcoef, mintercept = mmodel.coefs_, mmodel.intercepts_
    mmx, msx = mdict['norm']['mx'], mdict['norm']['sx']
    mmy, msy = mdict['norm']['my'], mdict['norm']['sy']

    posdata = [pmx, psx, pcoef, pintercept]
    mdata = [mmx, msx, mmy, msy, mcoef, mintercept]

    
    code.apply_halomodel(model = 'model', final='final', posdata=posdata, mdata=mdata, \
                         R1=R1, R2=R2)

    print('Checking gradient now')
    eps = 1e-4
    check_grad(code, 'model', 'final', init={'final': final}, eps=eps,
                rtol=1e-8, atol = 1e-6)

