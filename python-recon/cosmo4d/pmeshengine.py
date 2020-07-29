from __future__ import absolute_import

import numpy
from abopt.vmad2 import ZERO, Engine, statement, programme, CodeSegment, Literal
from abopt.abopt2 import VectorSpace
from pmesh.pm import ParticleMesh, RealField, ComplexField

def nyquist_mask(factor, v):
    # any nyquist modes are set to 0 if the transfer function is complex
    mask = (numpy.imag(factor) == 0) | \
            ~numpy.bitwise_and.reduce([(ii == 0) | (ii == ni // 2) for ii, ni in zip(v.i, v.Nmesh)])
    return factor * mask

class ParticleMeshVectorSpace(VectorSpace):
    def __init__(self, pm, q):
        self.qshape = q.shape
        self.pm = pm

    def addmul(self, a, b, c, p=1):
        if isinstance(b, RealField):
            r = b.copy()
            r[...] = a + b * c ** p
            return r
        elif isinstance(b, ComplexField):
            r = b.copy()
            if isinstance(c, ComplexField):
                c = c.plain
            if isinstance(a, ComplexField):
                a = a.plain
            r.plain[...] = a + b.plain * c ** p
            return r
        elif numpy.isscalar(b):
            return a + b * c ** p
        elif isinstance(b, numpy.ndarray):
            assert len(b) == self.qshape[0]
            return a + b * c ** p
        else:
            raise TypeError("type unknown")

    def dot(self, a, b):
        if type(a) != type(b):
            raise TypeError("type mismatch")

        if isinstance(a, RealField):
            return a.cdot(b)
        elif isinstance(a, ComplexField):
            return a.cdot(b)
        elif isinstance(a, numpy.ndarray):
            assert len(a) == len(b)
            assert len(a) == self.qshape[0]
            return self.pm.comm.allreduce(a.dot(b))
        else:
            raise TypeError("type unknown")

class ParticleMeshEngine(Engine):
    def __init__(self, pm, q=None):
        self.pm = pm
        if q is None:
            q = pm.generate_uniform_particle_grid(shift=0.0, dtype='f4')
        self.q = q
        self.vs = ParticleMeshVectorSpace(self.pm, self.q)

    @programme(ain=['s'], aout=['x'])
    def get_x(engine, s, x):
        code = CodeSegment(engine)
        code.add(x1='s', x2=Literal(engine.q), y='x')
        return code

    @statement(aout=['real'], ain=['complex'])
    def c2r(engine, real, complex):
        real[...] = complex.c2r()

    @c2r.defvjp
    def _(engine, _real, _complex):
        _complex[...] = _real.c2r_vjp()

    @c2r.defjvp
    def _(engine, real_, complex_):
        real_[...] = complex_.c2r()

    @statement(aout=['complex'], ain=['real'])
    def r2c(engine, complex, real):
        complex[...] = real.r2c()

    @r2c.defvjp
    def _(engine, _complex, _real):
        _real[...] = _complex.r2c_vjp()

    @r2c.defjvp
    def _(engine, complex_, real_):
        complex_[...] = real_.r2c()

    @statement(aout=['complex'], ain=['complex'])
    def decompress(engine, complex):
        return

    @decompress.defvjp
    def _(engine, _complex):
        _complex.decompress_vjp(out=Ellipsis)

    @decompress.defjvp
    def _(engine, complex_):
        pass # XXX: is this correct?

    @staticmethod
    def _lowpass_filter(k, v, Neff):
        k0s = 2 * numpy.pi / v.BoxSize
        mask = numpy.bitwise_and.reduce([abs(ki) <= Neff//2 * k0 for ki, k0 in zip(k, k0s)])
        return v * mask

    @statement(aout=['real'], ain=['real'])
    def lowpass(engine, real, Neff):
        real.r2c(out=Ellipsis).apply(
            lambda k, v: engine._lowpass_filter(k, v, Neff),
            out=Ellipsis).c2r(out=Ellipsis)

    @lowpass.defvjp
    def _(engine, _real, Neff):
        _real.c2r_vjp().apply(
            lambda k, v: engine._lowpass_filter(k, v, Neff),
            out=Ellipsis).r2c_vjp(out=Ellipsis)

    @lowpass.defjvp
    def _(engine, real_, Neff):
        real_.r2c().apply(
            lambda k, v: engine._lowpass_filter(k, v, Neff),
            out=Ellipsis).c2r(out=Ellipsis)

    @statement(aout=['layout'], ain=['x'])
    def decompose(engine, layout, x):
        pm = engine.pm
        layout[...] = pm.decompose(x)

    @decompose.defvjp
    def _(engine, _layout, _x):
        _x[...] = ZERO

    @decompose.defjvp
    def _(engine, layout_, x_):
        layout_[...] = ZERO

    @statement(aout=['mesh'], ain=['x', 'layout', 'mass'])
    def paint(engine, x, mesh, layout, mass=Literal(1.0)):
        pm = engine.pm
        N = pm.comm.allreduce(len(x))
        mesh[...] = pm.paint(x, mass=mass, layout=layout, hold=False)
        # to have 1 + \delta on the mesh
        mesh[...][...] *= 1.0 * pm.Nmesh.prod() / N

    @paint.defvjp
    def _(engine, _x, _mesh, x, mass, _mass, layout, _layout):
        pm = engine.pm
        _layout[...] = ZERO
        N = pm.comm.allreduce(len(x))
        _x[...], _mass[...] = pm.paint_vjp(_mesh, x, layout=layout, mass=mass)
        _x[...][...] *= 1.0 * pm.Nmesh.prod() / N
        _mass[...][...] *= 1.0 * pm.Nmesh.prod() / N

    @paint.defjvp
    def _(engine, x_, mesh_, x, layout, layout_, mass, mass_):
        pm = engine.pm
        if x_ is ZERO: x_ = None
        if mass_ is ZERO: mass_ = None # force cast it to a scale 0
        mesh_[...] = pm.paint_jvp(x, v_mass=mass_, mass=mass, v_pos=x_, layout=layout)

    @statement(aout=['mesh'], ain=['x', 'layout', 'mass'])
    def paintdirect(engine, x, mesh, layout, mass=Literal(1.0)):
        pm = engine.pm
        N = pm.comm.allreduce(len(x))
        mesh[...] = pm.paint(x, mass=mass, layout=layout, hold=False)

    @paintdirect.defvjp
    def _(engine, _x, _mesh, x, mass, _mass, layout, _layout):
        pm = engine.pm
        _layout[...] = ZERO
        N = pm.comm.allreduce(len(x))
        _x[...], _mass[...] = pm.paint_vjp(_mesh, x, layout=layout, mass=mass)

    @paintdirect.defjvp
    def _(engine, x_, mesh_, x, layout, layout_, mass, mass_):
        pm = engine.pm
        if x_ is ZERO: x_ = None
        if mass_ is ZERO: mass_ = None # force cast it to a scale 0
        mesh_[...] = pm.paint_jvp(x, v_mass=mass_, mass=mass, v_pos=x_, layout=layout)


    @statement(aout=['value'], ain=['x', 'mesh', 'layout'])
    def readout(engine, value, x, mesh, layout, resampler=None):
        pm = engine.pm
        N = pm.comm.allreduce(len(x))
        value[...] = mesh.readout(x, layout=layout, resampler=resampler)

    @readout.defvjp
    def _(engine, _value, _x, _mesh, x, layout, mesh, resampler):
        pm = engine.pm
        _mesh[...], _x[...] = mesh.readout_vjp(x, _value, layout=layout, resampler=resampler)

    @readout.defjvp
    def _(engine, value_, x_, mesh_, x, layout, mesh, layout_, resampler):
        pm = engine.pm
        if mesh_ is ZERO: mesh_ = None
        if x_ is ZERO: x_ = None
        value_[...] = mesh.readout_jvp(x, v_self=mesh_, v_pos=x_, layout=layout, resampler=resampler)

    @statement(aout=['complex'], ain=['complex'])
    def transfer(engine, complex, tf):
        complex.apply(lambda k, v: nyquist_mask(tf(k), v) * v, out=Ellipsis)
        
    @transfer.defvjp
    def _(engine, tf, _complex):
        _complex.apply(lambda k, v: nyquist_mask(numpy.conj(tf(k)), v) * v, out=Ellipsis)

    @transfer.defjvp
    def _(engine, tf, complex_):
        complex_.apply(lambda k, v: nyquist_mask(tf(k), v) * v, out=Ellipsis)

    @statement(aout=['residual'], ain=['model'])
    def residual(engine, model, data, sigma, residual):
        """
            residual = (model - data) / sigma

            J = 1 / sigma
        """
        residual[...] = (model - data) / sigma

    @residual.defvjp
    def _(engine, _model, _residual, data, sigma):
        _model[...] = _residual / sigma

    @residual.defjvp
    def _(engine, model_, residual_, data, sigma):
        residual_[...] = model_ / sigma

#    @statement(ain=['vec'], aout=['scalar'])
#    def vec1_to_scalar(engine, vec1, scalar):
#        tmp = 
#
#    @vec1_to_scalar.defvjp
#    def _(engine, _attribute, _value, dim):
#        _value[...] = _attribute[..., dim]
#
#    @vec1_to_scalar.defjvp
#    def _(engine, attribute_, value_, dim):
#        attribute_[..., dim] = value_
#
    @statement(ain=['attribute', 'value'], aout=['attribute'])
    def assign_component(engine, attribute, value, dim):
        attribute[..., dim] = value

    @assign_component.defvjp
    def _(engine, _attribute, _value, dim):
        _value[...] = _attribute[..., dim]

    @assign_component.defjvp
    def _(engine, attribute_, value_, dim):
        attribute_[..., dim] = value_

    @statement(ain=['attribute', 'value'], aout=['attribute'])
    def assign_chunk(engine, attribute, value, start, end):
        attribute[..., start:end] = value

    @assign_chunk.defvjp
    def _(engine, _attribute, _value, start, end):
        _value[...] = _attribute[..., start:end]

    @assign_chunk.defjvp
    def _(engine, attribute_, value_, start, end):
        attribute_[..., start:end] = value_

    @statement(ain=['x'], aout=['y'])
    def assign(engine, x, y):
        y[...] = x.copy()

    @assign.defvjp
    def _(engine, _y, _x):
        _x[...] = _y

    @assign.defjvp
    def _(engine, y_, x_, x):
        try:
            y_[...] = x.copy()
            y_[...][...] = x_
        except:
            y_[...] = x_

    @statement(ain=['x1', 'x2'], aout=['y'])
    def add(engine, x1, x2, y):
        y[...] = x1 + x2

    @add.defvjp
    def _(engine, _y, _x1, _x2):
        _x1[...] = _y
        _x2[...] = _y

    @add.defjvp
    def _(engine, y_, x1_, x2_):
        y_[...] = x1_ + x2_

    @statement(aout=['y'], ain=['x1', 'x2'])
    def multiply(engine, x1, x2, y):
        y[...] = x1 * x2

    @multiply.defvjp
    def _(engine, _x1, _x2, _y, x1, x2):
        _x1[...] = _y * x2
        _x2[...] = _y * x1

    @multiply.defjvp
    def _(engine, x1_, x2_, y_, x1, x2):
        y_[...] = x1_ * x2 + x1 * x2_


    @statement(aout=['y'], ain=['x1', 'x2'])
    def divide(engine, x1, x2, y):
        y[...] = x1 / x2

    @divide.defvjp
    def _(engine, _x1, _x2, _y, x1, x2):
        _x1[...] = _y / x2
        _x2[...] = _y * x1 / x2**2 *-1

    @divide.defjvp
    def _(engine, x1_, x2_, y_, x1, x2):
        y_[...] = x1_ / x2 - x1 / x2**2 * x2_


    @statement(aout=['y'], ain=['x'])
    def matrix_cmul(engine, x, y, W):
        y[...] = numpy.dot(x, W)

    @matrix_cmul.defvjp
    def _(engine, _x, _y, W):
        _x[...] = numpy.dot(_y, W.T)

    @matrix_cmul.defjvp
    def _(engine, x_, y_, W):
        y_[...] = numpy.dot(x_, W)


    @statement(ain=['x'], aout=['y'])
    def to_scalar(engine, x, y):
        if isinstance(x, RealField):
            y[...] = x.cnorm()
        elif isinstance(x, ComplexField):
            raise TypeError("Computing the L-2 norm of complex is not a good idea, because the gradient propagation is ambiguous")
        else:
            y[...] = engine.pm.comm.allreduce((x[...] ** 2).sum(dtype='f8'))

    @to_scalar.defvjp
    def _(engine, _y, _x, x):
        _x[...] = x * (2 * _y)

    @to_scalar.defjvp
    def _(engine, y_, x_, x):
        if isinstance(x, RealField):
            y_[...] = x.cdot(x_) * 2
        elif isinstance(x, ComplexField):
            raise TypeError("Computing the L-2 norm of complex is not a good idea, because the gradient propagation is ambiguous")
        else:
            y_[...] = engine.pm.comm.allreduce((x * x_).sum(dtype='f8')) * 2


    @statement(ain=['x'], aout=['y'])
    def L1norm(engine, x, y):
        if isinstance(x, RealField):
            y[...] = abs(x).csum()
        elif isinstance(x, ComplexField):
            raise TypeError("Computing the L-1 norm of complex is not a good idea")
        else:
            y[...] = engine.pm.comm.allreduce(abs(x[...]).sum(dtype='f8'))

    @L1norm.defvjp
    def _(engine, _y, _x, x):
        _x[...] = x.copy()
        _x[...][...] = _y * numpy.sign(x)
        #print(type(_y), type(numpy.sign(x)), type(_y * numpy.sign(x)))

    @L1norm.defjvp
    def _(engine, y_, x_, x):
        if isinstance(x, RealField):
            y_[...] = ((x_) * numpy.sign(x)).csum()
        elif isinstance(x, ComplexField):
            raise TypeError("Computing the L-1 norm of complex is not a good idea, because the gradient propagation is ambiguous")
        else:
            y_[...] = engine.pm.comm.allreduce((numpy.sign(x) * x_).sum(dtype='f8')) 
            #y_[...] = engine.pm.comm.allreduce((x_).sum(dtype='f8')) 



    @statement(ain=['x'], aout=['y'])
    def total(engine, x, y):
        if isinstance(x, RealField):
            y[...] = x.csum()
        elif isinstance(x, ComplexField):
            raise TypeError("Computing the total of complex is not a good idea")
        else:
            y[...] = engine.pm.comm.allreduce((x[...]).sum(dtype='f8'))

    @total.defvjp
    def _(engine, _y, _x, x):
        _x[...] = x.copy()
        _x[...][...] = _y 
        #print(type(_y), type(numpy.sign(x)), type(_y * numpy.sign(x)))

    @total.defjvp
    def _(engine, y_, x_, x):
        if isinstance(x, RealField):
            y_[...] = ((x_) ).csum()
        elif isinstance(x, ComplexField):
            raise TypeError("Computing the L-1 norm of complex is not a good idea, because the gradient propagation is ambiguous")
        else:
            y_[...] = engine.pm.comm.allreduce((x_).sum(dtype='f8')) 
            #y_[...] = engine.pm.comm.allreduce((x_).sum(dtype='f8')) 


def check_grad(code, yname, xname, init, eps, rtol, atol=1e-12, verbose=False, toscalar=True):
    from numpy.testing import assert_allclose
    engine = code.engine
    comm = engine.pm.comm
    if isinstance(init[xname], numpy.ndarray):
        x = init[xname]
        if x.ndim == 2:
            cshape = engine.pm.comm.allreduce(x.shape[0]), x.shape[1]
        else:
            cshape = engine.pm.comm.allreduce(x.shape[0]),

        def cperturb(pos, ind, eps):
            pos = pos.copy()
            start = sum(comm.allgather(pos.shape[0])[:comm.rank])
            end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
            if ind[0] >= start and ind[0] < end:
                ind1 = tuple([ind[i] - start if i == 0 else ind[i] for i in range(len(ind))])
                old = pos[ind1]
                coord = pos[ind[0]-start].copy()
                pos[ind1] = old + eps
                new = pos[ind1]
            else:
                old, new, coord = 0, 0, 0
            diff = comm.allreduce(new - old)
            return pos

        def cget(pos, ind):
            if pos is ZERO: return 0
            start = sum(comm.allgather(pos.shape[0])[:comm.rank])
            end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
            if ind[0] >= start and ind[0] < end:
                ind1 = tuple([ind[i] - start if i == 0 else ind[i] for i in range(len(ind))])
                old = pos[ind1]
            else:
                old = 0
            return comm.allreduce(old)

    elif isinstance(init[xname], RealField):
        cshape = init[xname].cshape
        def cget(real, index):
            if real is ZERO: return 0
            return real.cgetitem(index)

        def cperturb(real, index, eps):
            old = real.cgetitem(index)
            r1 = real.copy()
            r1.csetitem(index, old + eps)
            return r1

    code = code.copy()
    if toscalar:
        code.to_scalar(x=yname, y='y')
    else:
        code.assign(x=yname, y='y')

    y, tape = code.compute('y', init=init, return_tape=True)
    vjp = tape.get_vjp()
    jvp = tape.get_jvp()

    _x = vjp.compute('_' + xname, init={'_y' : 1.0})

    center = init[xname]
    init2 = init.copy()
    ng_bg = []
    fg_bg = []
    for index in numpy.ndindex(*cshape):
        x1 = cperturb(center, index, eps)
        x0 = cperturb(center, index, -eps)
        analytic = cget(_x, index)
        init2[xname] = x1
        y1 = code.compute('y', init2)
        init2[xname] = x0
        y0 = code.compute('y', init2)

        base = (x1 - x0)
        y_ = jvp.compute('y_', init={xname + '_': base})

        #logger.DEBUG("CHECKGRAD: %s" % (y1, y0, y1 - y0, get_pos(code.engine, _x, index) * 2 * eps))
        if verbose:
            print("CHECKGRAD: ", index, (x1 - x0)[...].max(), y, y1 - y0, y_, cget(_x, index) * 2 * eps)

        fg_bg.append([index, y_, cget(_x, index) * 2 * eps])

        ng_bg.append([index, y1 - y0, cget(_x, index) * 2 * eps])

    fg_bg = numpy.array(fg_bg, dtype='O')
    ng_bg = numpy.array(ng_bg, dtype='O')

    def errorstat(stat, rtol, atol):
        g1 = numpy.array([a[1] for a in stat])
        g2 = numpy.array([a[2] for a in stat])

        ag1 = abs(g1) + (abs(g1) == 0) * numpy.std(g1)
        ag2 = abs(g2) + (abs(g2) == 0) * numpy.std(g2)
        sig = (g1 - g2) / ((ag1 + ag2) * rtol + atol)
        bins = [-100, -50, -20, -1, 1, 20, 50, 100]
        d = numpy.digitize(sig, bins)
        return d

    d1 = errorstat(fg_bg, rtol, atol)

    d2 = errorstat(ng_bg, rtol * 10, atol)

    


    if (d1 != 4).any():
        print('ngbg = ', ng_bg)
        print('fgbg = ', fg_bg)
        #print('ngbg = ' ng_bg)
        raise AssertionError("FG_BG Bad gradients: %s " % numpy.bincount(d1))


    if (d2 != 4).any():
        raise AssertionError("NG_BG Bad gradients: %s " % numpy.bincount(d2))

