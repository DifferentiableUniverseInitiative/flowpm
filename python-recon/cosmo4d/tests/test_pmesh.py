from __future__ import print_function
from __future__ import absolute_import

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
import numpy
import logging

from pmesh.pm import ParticleMesh
pm = ParticleMesh(BoxSize=1.0, Nmesh=(8, 8, 8), dtype='f8', resampler='cic')

from abopt.vmad2 import CodeSegment, logger

from cosmo4d.pmeshengine import ParticleMeshEngine,  check_grad, Literal

logger.setLevel(level=logging.WARNING)

def test_compute():
    def transfer(k): return 2.0
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')
    code.to_scalar(x='r', y='sum')

    field = pm.generate_whitenoise(seed=1234).c2r()

    norm = code.compute('sum', init={'r': field})
    assert_allclose(norm, field.cnorm() * 4)

def test_vs():
    engine = ParticleMeshEngine(pm)
    field = pm.generate_whitenoise(seed=1234)
    a = engine.vs.zeros_like(field)
    assert_allclose(a, 0)

def test_vjp():
    def transfer(k):
        k2 = sum(ki **2 for ki in k)
        k2[k2 == 0] = 1.0
#        return 1 / k2
        return 2.0
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')
    code.multiply(x1='r', x2=Literal(0.1), y='r')
    code.to_scalar(x='r', y='sum')

    field = pm.generate_whitenoise(seed=1234).c2r()

    norm, tape = code.compute('sum', init={'r': field}, return_tape=True)
    assert_allclose(norm, field.cnorm() * 4 * 0.1 ** 2)

    vjp = tape.get_vjp()
    _r = vjp.compute('_r', init={'_sum': 1.0})
    assert_allclose(_r, field * 4 * 2 * 0.1 * 0.1)

def test_to_scalar():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    numpy.random.seed(1234)
    s = numpy.random.uniform(size=engine.q.shape) * 0.1
    check_grad(code, 's', 's', init={'s': s}, eps=1e-4, rtol=1e-8)

def test_L1norm():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    numpy.random.seed(1234)
    code.L1norm(x='s', y='l')

    #s = numpy.random.uniform(1, 2, size=engine.q.shape)
    #s = numpy.ones_like(engine.q)
    s = pm.generate_whitenoise(seed=1234, mode='real')
    
    l = code.compute(['l'], init={'s':s})
    print(l)
    check_grad(code, 'l', 's', init={'s': s}, eps=1e-8, rtol=1e-4,  toscalar=False)

def test_total():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    numpy.random.seed(1234)
    code.total(x='s', y='l')

    #s = numpy.random.uniform(1, 2, size=engine.q.shape)
    #s = numpy.ones_like(engine.q)
    s = pm.generate_whitenoise(seed=1234, mode='real')
    
    l = code.compute(['l'], init={'s':s})
    print(l)
    check_grad(code, 'l', 's', init={'s': s}, eps=1e-8, rtol=1e-4,  toscalar=False)



def test_paint():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    s = pm.BoxSize / pm.Nmesh * 0.001 + 0.99 * engine.q / pm.Nmesh # sample all positions.
    m = numpy.ones(len(engine.q)) * 3

    code.get_x(s='s', x='x')
    code.decompose(x='x', layout='layout')
    code.paint(x='x', mesh='density', layout='layout', mass='m')

    check_grad(code, 'density', 's', init={'s': s, 'm' : m}, eps=1e-4, rtol=1e-8, atol=1e-11)
    check_grad(code, 'density', 'm', init={'s': s, 'm' : m}, eps=1e-4, rtol=1e-8, atol=1e-11)


def test_paintdirect():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    s = pm.BoxSize / pm.Nmesh * 0.001 + 0.99 * engine.q / pm.Nmesh # sample all positions.
    m = numpy.ones(len(engine.q)) * 3

    code.get_x(s='s', x='x')
    code.decompose(x='x', layout='layout')
    code.paintdirect(x='x', mesh='density', layout='layout', mass='m')

    check_grad(code, 'density', 's', init={'s': s, 'm' : m}, eps=1e-4, rtol=1e-8, atol=1e-11)
    check_grad(code, 'density', 'm', init={'s': s, 'm' : m}, eps=1e-4, rtol=1e-8, atol=1e-11)


def test_readout():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    s = pm.BoxSize / pm.Nmesh * 0.001 + 0.99 * engine.q / pm.Nmesh # sample all positions.

    field = pm.generate_whitenoise(seed=1234, mode='real')

    code.get_x(s='s', x='x')
    code.decompose(x='x', layout='layout')
    code.readout(x='x', mesh='density', layout='layout', value='value')

    check_grad(code, 'value', 'density', init={'density' : field, 's': s}, eps=1e-4, rtol=1e-8)

    check_grad(code, 'value', 's', init={'density' : field, 's': s}, eps=1e-4, rtol=1e-8)


def test_transfer_imag():
    def transfer(k):
        return 1j * k[0]

    field = pm.generate_whitenoise(seed=1234, mode='real')

    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(complex='c', real='r')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')

    check_grad(code, 'r', 'r', init={'r': field}, eps=1e-4, rtol=1e-8)

def test_transfer_real():
    def transfer(k):
        return k[0]

    field = pm.generate_whitenoise(seed=1234, mode='real')

    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(complex='c', real='r')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')

    check_grad(code, 'r', 'r', init={'r': field}, eps=1e-4, rtol=1e-8)

def test_c2rr2c():
    field = pm.generate_whitenoise(seed=1234, mode='real')

    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.c2r(complex='c', real='r')

    check_grad(code, 'r', 'r', init={'r': field}, eps=1e-4, rtol=1e-8)

def test_lowpass():
    field = pm.generate_whitenoise(seed=1234, mode='real')

    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.lowpass(real='r', Neff=1)

    check_grad(code, 'r', 'r', init={'r': field}, eps=1e-4, rtol=1e-8)
