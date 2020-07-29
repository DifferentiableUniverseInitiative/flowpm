from __future__ import print_function
import numpy
import logging

from .pmeshengine import (
        ParticleMeshEngine,
        ZERO, Literal,
        CodeSegment,
        programme,
        statement,
        ParticleMesh, RealField, ComplexField
        )

class FastPMEngine(ParticleMeshEngine):
    def __init__(self, pm, B=1, shift=0.25):
        ParticleMeshEngine.__init__(self, pm, pm.generate_uniform_particle_grid(shift=0.0))

        # force pm is higher resolution than the particle pm.
        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.fengine = ParticleMeshEngine(fpm, q=self.q)

    @programme(ain=['whitenoise'], aout=['dlinear_k'])
    def create_linear_field(engine, whitenoise, powerspectrum, dlinear_k):
        """ Generate a linear over-density field

            Parameters
            ----------
            whitenoise : RealField, in
                the white noise field
            powerspectrum : function, ex
                P(k) in Mpc/h units.
            dlinear_k : ComplexField, out
                the over-density field in fourier space
        """
        code = CodeSegment(engine)
        code.r2c(real=whitenoise, complex=dlinear_k)
        def tf(k):
            k2 = sum(ki**2 for ki in k)
            r = (powerspectrum(k2 ** 0.5) / engine.pm.BoxSize.prod()) ** 0.5
            r[k2 == 0] = 1.0
            return r
        code.transfer(complex=dlinear_k, tf=tf)
        return code

    @programme(aout=['whitenoise'], ain=['dlinear_k'])
    def create_whitenoise(engine, whitenoise, powerspectrum, dlinear_k):
        """ Generate a whitenoise field

            Parameters
            ----------
            whitenoise : RealField, out
                the white noise field
            powerspectrum : function, ex
                P(k) in Mpc/h units.
            dlinear_k : ComplexField, in
                the over-density field in fourier space
        """
        code = CodeSegment(engine)
        def tf(k):
            k2 = sum(ki**2 for ki in k)
            r = (powerspectrum(k2 ** 0.5) / engine.pm.BoxSize.prod()) ** -0.5
            r[k2 == 0] = 1.0
            return r
        code.assign(x='dlinear_k', y='tmp')
        code.transfer(complex='tmp', tf=tf)
        code.c2r(real=whitenoise, complex='tmp')
        return code

    @programme(aout=['smoothed'], ain=['d_k'])
    def gauss_smoothing(engine, smoothed, R, d_k):
        """ Smooth the field with Gaussian filter

            Parameters
            ----------
            gauss : RealField, out
                the Gaussian smoothed field
            R : float
                Radius of the Gaussian kernel
            d_k : ComplexField, in
                the input field in fourier space
        """
        code = CodeSegment(engine)
        def tf(k):
            k2 = sum(ki**2 for ki in k)
            wts = numpy.exp(-0.5*k2* R**2)
            return wts
            
        code.assign(x='d_k', y='tmp')
        code.transfer(complex='tmp', tf=tf)
        code.c2r(real=smoothed, complex='tmp')
        return code

    @programme(aout=['smoothed'], ain=['d_k'])
    def fingauss_smoothing(engine, smoothed, R, d_k):
        """ Smooth the field with Gaussian filter

            Parameters
            ----------
            gauss : RealField, out
                the Gaussian smoothed field
            R : float
                Radius of the Gaussian kernel
            d_k : ComplexField, in
                the input field in fourier space
        """
        code = CodeSegment(engine)
        def tf(k):
            k2 = sum(((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny)))**2  for ki in k)
            wts = numpy.exp(-0.5*k2* R**2)
            return wts
            
        kny = numpy.pi*engine.pm.Nmesh[0]/engine.pm.BoxSize[0]
        code.assign(x='d_k', y='tmp')
        code.transfer(complex='tmp', tf=tf)
        code.c2r(real=smoothed, complex='tmp')
        return code

    @programme(aout=['deconvolved'], ain=['d_k'])
    def de_cic(engine, deconvolved,  d_k):
        """ Smooth the field with Gaussian filter

            Parameters
            ----------
            gauss : RealField, out
                the Gaussian smoothed field
            R : float
                Radius of the Gaussian kernel
            d_k : ComplexField, in
                the input field in fourier space
        """
        code = CodeSegment(engine)
        def tf(k):
            kny = [numpy.sinc(k[i]*engine.pm.BoxSize[i]/(2*numpy.pi*engine.pm.Nmesh[i])) for i in range(3)]
            wts = (kny[0]*kny[1]*kny[2])**-2
            return wts
            
        #kny = numpy.pi*engine.pm.Nmesh[0]/engine.pm.BoxSize[0]
        code.assign(x='d_k', y='tmp')
        code.transfer(complex='tmp', tf=tf)
        code.c2r(real=deconvolved, complex='tmp')
        return code


    @programme(ain=['source_k'], aout=['s'])
    def solve_linear_displacement(engine, source_k, s):
        """ Solve linear order displacement from a source.

            Parameters
            ----------
            source_k: ComplexField, in
                the source, over-density. Zero mode is neglected.
            x : array, out
                linear position induced by the source.
        """
        code = CodeSegment(engine)
        code.assign(x=Literal(engine.q), y='q')
        code.assign(x=Literal(numpy.zeros_like(engine.q)), y=s)
        code.decompose(x='q', layout='layout')
        for d in range(engine.pm.ndim):
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[d] / k2 * ~mask
            code.assign(x='source_k', y='disp1_k')
            code.transfer(complex='disp1_k', tf=tf)
            code.c2r(complex='disp1_k', real='disp1')
            code.readout(mesh='disp1', value='s1', x='q', layout='layout')
            code.assign_component(attribute=s, value='s1', dim=d)
        return code

    @statement(ain=['x1', 'x2'], aout='y')
    def bilinear(engine, x1, c1, x2, c2, y):
        y[...] = x1 * c1 + x2 * c2

    @bilinear.defvjp
    def _(engine, _x1, _x2, _y, c1, c2):
        _x1[...] = _y * c1
        _x2[...] = _y * c2

    @bilinear.defjvp
    def _(engine, x1_, x2_, y_, c1, c2):
        y_[...] = x1_ * c1 + x2_ * c2

    @programme(ain=['dlinear_k'], aout=['s', 'v', 's1'])
    def solve_za(engine, pt, aend, dlinear_k, s, v, s1):
        """ Solve N-body with Lagrangian perturbation theory

            Parameters
            ----------
            dlinear_k: ComplexField, in
                linear overdensity field
            s : Array, out
                displaement of particles
            v : Array, out
                conjugate momentum of particles, a**2 H0 v_pec
            s1 : Array, out
                First order LPT displacement
        """
        code = CodeSegment(engine)
        code.solve_linear_displacement(source_k='dlinear_k', s=s1)
        code.multiply(x1='s1', x2=Literal(pt.D1(aend)), y='s')
        code.multiply(x1='s1', x2=Literal(pt.f1(aend) * aend ** 2 * pt.E(aend) * pt.D1(aend)), y='v')
        return code

    @programme(ain=['dlinear_k'], aout=['s', 'v', 's1', 's2'])
    def solve_lpt(engine, pt, aend, dlinear_k, s, v, s1, s2):
        """ Solve N-body with Lagrangian perturbation theory

            Parameters
            ----------
            dlinear_k: ComplexField, in
                linear overdensity field
            s : Array, out
                displaement of particles
            v : Array, out
                conjugate momentum of particles, a**2 H0 v_pec
            s1 : Array, out
                First order LPT displacement
            s2 : Array, out
                Second order LPT displacement
        """
        code = CodeSegment(engine)
        code.solve_linear_displacement(source_k='dlinear_k', s=s1)
        code.generate_2nd_order_source(source_k='dlinear_k', source2_k='source2_k')
        code.solve_linear_displacement(source_k='source2_k', s=s2)

        code.bilinear(x1='s1', c1=pt.D1(aend),
                      x2='s2', c2=pt.D2(aend),
                       y=s)

        code.bilinear(x1='s1', c1=pt.f1(aend) * aend ** 2 * pt.E(aend) * pt.D1(aend),
                      x2='s2', c2=pt.f2(aend) * aend ** 2 * pt.E(aend) * pt.D2(aend),
                       y=v)
        return code

    @programme(ain=['field'], aout=['features'])
    def find_neighbours(engine, field, features):
        """ Find neighbours of a field.

            This will create a vector for each position in the field,
            containing the nearest neighbours in a 3x3x3 matrix
        """
        code = CodeSegment(engine)
        N = len(engine.q)
        Nf = 3 ** engine.pm.ndim
        code.assign(x=Literal(numpy.zeros((N, Nf))), y='features')
        grid = engine.pm.generate_uniform_particle_grid(shift=0)
        for i in range(Nf):
            ii = i
            a = []
            for d in range(engine.pm.ndim):
                a.append(ii % 3 - 1)
                ii //= 3

            grid1 = grid + numpy.array(a[::-1]) * (engine.pm.BoxSize / engine.pm.Nmesh)
            layout = engine.pm.decompose(grid1)
            code.readout(x=Literal(grid1), mesh='field', value='feature1', layout=Literal(layout), resampler='nearest')
            code.assign_component(attribute='features', value='feature1', dim=i)
        return code
##

    @programme(ain=['features'], aout=['predict'])
    def apply_nets(engine, predict, features, arch, Nd, t=0, w=1):
        """ Apply a trained neural network to predict

            This will return a vector of predictions based on the input features and
        and the weights and intercepts of the trained Neural Network
        """
        code = CodeSegment(engine)
        code.assign(x='features', y='inp')
        ndim = Nd
        for i in arch:
            W, b, f = i
            code.assign(x=Literal(numpy.zeros((ndim, W.shape[1]))), y='tmp')            
            code.matrix_cmul(W=W, x='inp', y='Winp')
            code.add(x1='Winp', x2=Literal(b), y='bWinp')
            if f == 'relu':
                code.relu(x='bWinp', y='inp')
            if f == 'elu':
                code.elu(x='bWinp', y='inp')
            elif f == 'linear':
                code.identity(x='bWinp', y='inp')
            elif f == 'sigmoid':
                code.logistic(x='bWinp', y='inp', t=t, w=w)
        code.assign(x='inp', y='predict')
        return code


    @statement(aout=['y'], ain=['x'])
    def relu(engine, x, y):
        tmp = x.copy()
        tmp[x[...]<0] = 0
        y[...] = tmp

    @relu.defvjp
    def _(engine, _x, _y, x):
        tmp = _y.copy()
        tmp[x[...]<0] = 0
        _x[...] = tmp

    @relu.defjvp
    def _(engine, x_, y_, x):
        tmp = x_.copy()
        tmp[x[...]<0] = 0
        y_[...] = tmp

    @statement(aout=['y'], ain=['x'])
    def elu(engine, x, y, alpha=1):
        tmp = x.copy()
        tmp[x[...]<0] = alpha*(numpy.exp(x[x<0]) - 1)
        y[...] = tmp

    @elu.defvjp
    def _(engine, _x, _y, x, alpha=1):
        tmp = _y.copy()
        tmp[x[...]<0] = alpha*(numpy.exp(x[x<0]))*tmp[x<0]
        _x[...] = tmp

    @elu.defjvp
    def _(engine, x_, y_, x, alpha=1):
        tmp = x_.copy()
        tmp[x[...]<0] = alpha*(numpy.exp(x[x<0]))*tmp[x<0]
        y_[...] = tmp


    @statement(aout=['y'], ain=['x'])
    def identity(engine, x, y):
        tmp = x.copy()
        y[...] = tmp

    @identity.defvjp
    def _(engine, _x, _y, x):
        tmp = _y.copy()
        _x[...] = tmp

    @identity.defjvp
    def _(engine, x_, y_, x):
        tmp = x_.copy()
        y_[...] = tmp


    @statement(aout=['y'], ain=['x'])
    def logistic(engine, x, y, t, w):
        tmp = x.copy()
        wts = 1 + numpy.exp(-w*(x-t))
        wts = 1/wts
        tmp[...] = wts
        y[...] = tmp

    @logistic.defvjp
    def _(engine, _x, _y, x, t, w):
        tmp = x.copy()
        wts = 1 + numpy.exp(-w*(x-t))
        wts = 1/wts
        wts = w*wts*(1-wts)*_y
        tmp[...] = wts
        _x[...] = tmp

    @logistic.defjvp
    def _(engine, x_, y_, x, t, w):
        tmp = x.copy()
        wts = 1 + numpy.exp(-w*(x-t))
        wts = 1/wts
        wts = w*wts*(1-wts)*x_
        tmp[...] = wts
        y_[...] = tmp

    @statement(aout=['y'], ain=['x'])
    def threshold(engine, x, y, t):
        tmp = x.copy()
        tmp[tmp<t] = 0
        tmp[tmp>=t] = 1
        y[...] = tmp

    @threshold.defvjp
    def _(engine, _x, _y, x, t):
        _x[...] = ZERO

    @threshold.defjvp
    def _(engine, x_, y_, x, t):
        y_[...] = ZERO


    @statement(aout=['y'], ain=['x'])
    def reshape_scalar(engine, x, y):
        y[...] = x.reshape(-1)

    @reshape_scalar.defvjp
    def _(engine, _x, _y, x):
        _x[...] = _y.reshape(x.shape)

    @reshape_scalar.defjvp
    def _(engine, x_, y_):
        y_[...] = x_.reshape(-1)

    @programme(ain=['dlinear_k'], aout=['s', 'v', 's1', 's2'])
    def solve_fastpm(engine, pt, asteps, dlinear_k, s, v, s1, s2, order=2):
        """ Solve N-body with FastPM

            Parameters
            ----------
            dlinear_k: ComplexField, in
                linear overdensity field
            asteps : Array, 1d, ex
                time steps. LPT is used to solve asteps[0], then a KDK scheme is used
                to evolve the field to asteps[-1]
            s : Array, out
                displaement of particles
            v : Array, out
                conjugate momentum of particles, a**2 H0 v_pec
            s1 : Array, out
                First order LPT displacement
            s2 : Array, out
                Second order LPT displacement
        """
        code = CodeSegment(engine)
        if order == 2: code.solve_lpt(pt=pt, aend=asteps[0], dlinear_k=dlinear_k, s=s, v=v, s1=s1, s2=s2)
        elif order == 1:
            code.solve_za(pt=pt, aend=asteps[0], dlinear_k=dlinear_k, s=s, v=v, s1=s1)
            code.multiply(x1=s1, x2=Literal(0), y=s2)

        def K(ai, af, ar):
            return 1 / (ar ** 2 * pt.E(ar)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ar)
        def D(ai, af, ar):
            return 1 / (ar ** 3 * pt.E(ar)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ar)

        code.assign(x=Literal(numpy.zeros_like(engine.q)), y='f')

        code.force(s=s, force='f', force_factor=1.5 * pt.Om0)
        for ai, af in zip(asteps[:-1], asteps[1:]):
            ac = (ai * af) ** 0.5
            code.kick(v=v, f='f', kick_factor=K(ai, ac, ai))
            code.drift(x=s, v=v, drift_factor=D(ai, ac, ac))
            code.drift(x=s, v=v, drift_factor=D(ac, af, ac))
            code.force(s=s, force='f', force_factor=1.5 * pt.Om0)
            code.kick(v=v, f='f', kick_factor=K(ac, af, af))
        return code

    @programme(ain=['source_k'], aout=['source2_k'])
    def generate_2nd_order_source(engine, source_k, source2_k):
        """ Generate 2nd order LPT source from 1st order LPT source

        """
        code = CodeSegment(engine)
        if engine.pm.ndim < 3:
            code.assign(x=Literal(engine.pm.create(mode='complex', value=0)), y='source2_k')
            return code

        code.assign(x=Literal(engine.pm.create(mode='real', value=0)), y='source2')

        D1 = [1, 2, 0]
        D2 = [2, 0, 1]
        varname = ['var_%d' % d for d in range(engine.pm.ndim)]
        for d in range(engine.pm.ndim):
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[d] * 1j * k[d] / k2 * ~mask
            code.assign(x='source_k', y=varname[d])
            code.transfer(complex=varname[d], tf=tf)
            code.c2r(complex=varname[d], real=varname[d])

        for d in range(engine.pm.ndim):
            code.multiply(x1=varname[D1[d]], x2=varname[D2[d]], y='phi_ii')
            code.add(x1='source2', x2='phi_ii', y='source2')

        for d in range(engine.pm.ndim):
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[D1[d]] * 1j * k[D2[d]] / k2 * ~mask
            code.assign(x='source_k', y='phi_ij')
            code.transfer(complex='phi_ij', tf=tf)
            code.c2r(complex='phi_ij', real='phi_ij')
            code.multiply(x1='phi_ij', x2='phi_ij', y='phi_ij')
            code.multiply(x1='phi_ij', x2=Literal(-1.0), y='phi_ij')
            code.add(x1='source2', x2='phi_ij', y='source2')

        code.multiply(x1='source2', x2=Literal(3.0 /7), y='source2')
        code.r2c(real='source2', complex='source2_k')
        return code

    @programme(aout=['force'], ain=['s'])
    def force(engine, force, s, force_factor):
        """ Compute gravity force on paticles.

            Parameters
            ----------
            force : array, out
                force of particles
            s : array, in
                displacement of particles, producing a density field
            force_factor : float, ex
                 usually 1.5 * Om0, the scaling of the force

        """
        code = CodeSegment(engine)
        code.get_x(s=s, x='x')
        code.force_prepare(x='x', density_k='density_k', layout='layout')
        code.force_compute(x='x', density_k='density_k', layout='layout', force=force,
                force_factor=force_factor)
        return code

    @programme(aout=['density'], ain=['x'])
    def paint_simple(engine, density, x):
        """ Paint particles to a mesh with proper domain decomposition

        """
        code = CodeSegment(engine)
        code.decompose(x=x, layout='layout')
        code.paint(x=x, layout='layout', mesh=density)
        return code

    @programme(aout=['value'], ain=['x', 'mesh'])
    def readout_simple(engine, mesh, x, value):
        """ Paint particles to a mesh with proper domain decomposition

        """
        code = CodeSegment(engine)
        code.decompose(x=x, layout='layout')
        code.readout(x=x, layout='layout', mesh=mesh, value=value)
        return code

    @programme(aout=['density_k', 'layout'], ain=['x'])
    def force_prepare(engine, density_k, x, layout):
        code = CodeSegment(engine.fengine)
        code.decompose(x=x, layout=layout)
        code.paint(x=x, layout=layout, mesh='density')
        code.r2c(complex=density_k, real='density')
        return code

    @programme(aout=['force'], ain=['density_k', 'x', 'layout'])
    def force_compute(engine, force, density_k, x, layout, force_factor):
        code = CodeSegment(engine.fengine)
        code.assign(x=Literal(numpy.zeros_like(engine.q)), y='force')

        for d in range(engine.pm.ndim):
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return 1j * k[d] / k2 * ~mask
            code.assign(x='density_k', y='complex')
            code.transfer(complex='complex', tf=tf)
            code.c2r(complex='complex', real='real')
            code.readout(value='force1', mesh='real', x=x, layout='layout')
            code.assign_component(attribute='force', dim=d, value='force1')
        code.multiply(x1='force', x2=Literal(force_factor), y='force')

        return code


    @programme(ain=['source'], aout=['shear'])
    def generate_shear(engine, source, shear):
        """ Generate shear field
        """
        code = CodeSegment(engine)

        code.assign(x=Literal(engine.pm.create(mode='real', value=0)), y='shear')
        code.r2c(real='source', complex='source_k')
        code.c2r(complex='source_k', real='shear')

        for d in range(engine.pm.ndim):
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return  k[d]  * k[d] / k2 * ~mask - 1/3.
            code.assign(x='source_k', y='phi_ii')
            code.transfer(complex='phi_ii', tf=tf)
            code.c2r(complex='phi_ii', real='phi_ii')
            code.multiply(x1='phi_ii', x2='phi_ii', y='phi_ii')
            code.add(x1='shear', x2='phi_ii', y='shear')


        D1 = [1, 2, 0]
        D2 = [2, 0, 1]
        #offdiagonal
        for d in range(engine.pm.ndim):
            def tf(k, d=d):
                k2 = sum(ki ** 2 for ki in k)
                mask = k2 == 0
                k2[mask] = 1.0
                return k[D1[d]] * k[D2[d]] / k2 * ~mask 
            code.assign(x='source_k', y='phi_ij')
            code.transfer(complex='phi_ij', tf=tf)
            code.c2r(complex='phi_ij', real='phi_ij')
            code.multiply(x1='phi_ij', x2='phi_ij', y='phi_ij')
            code.multiply(x1='phi_ij', x2=Literal(2), y='phi_ij') #symmetry
            code.add(x1='shear', x2='phi_ij', y='shear')

        return code



    @statement(aout=['projection'], ain=['field'])
    def project(engine, projection, field):
        pm2d = engine.pm.resize((engine.pm.Nmesh[0], engine.pm.Nmesh[1], 1))
        p = pm2d.create(mode='real')
        p[...] = field[...].mean(axis=-1)[..., None]
        projection[...] = p

    @project.defvjp
    def _(engine, _projection, _field, field):
        _field[...] = field * 0.0 + _projection / engine.pm.Nmesh[-1]

    @project.defjvp
    def _(engine, projection_, field_, field):
        pm2d = engine.pm.resize((engine.pm.Nmesh[0], engine.pm.Nmesh[1], 1))
        p = pm2d.create(mode='real')
        p[...] = field_[...].mean(axis=-1)[..., None]
        projection_[...] = p

    @statement(aout=['y'], ain=['x'])
    def pow(engine, x, power, y):
        y[...] = x ** power

    @pow.defvjp
    def _(engine, _y, _x, x, power):
        if power != 1:
            _x[...] = _y * (x ** (power - 1))*power
        else:
            _x[...] = _y

    @pow.defjvp
    def _(engine, y_, x_, x, power):
        if power != 1:
            y_[...] = x_ * (x ** (power - 1))*power
        else:
            y_[...] = x_
##
    @statement(aout=['y'], ain=['x'])
    def log(engine, x, y):
        if numpy.isscalar(x):
            y[...] = numpy.log(x)
        else:
            y[...] = x.copy()
            y[...][...] = numpy.log(x)


    @log.defvjp
    def _(engine, _y, _x, x, ):
        if numpy.isscalar(x):
            _x[...] = _y * (x ** -1)
        else:
            _x[...] = _y.copy()
            _x[...][...] = _y * (x ** -1)

    @log.defjvp
    def _(engine, y_, x_, x,):
        if numpy.isscalar(x):
            y_[...] = x_ * (x ** -1)
        else:
            y_[...] = x_[...]
            y_[...][...] = x_ * (x ** -1)

    @statement(aout=['y'], ain=['x'])
    def expon(engine, x, y):
        y[...] = numpy.exp(x)

    @expon.defvjp
    def _(engine, _x, _y, x):
        _x[...] = _y.copy()
        _x[...][...] = _y*numpy.exp(x)

    @expon.defjvp
    def _(engine, x_, y_, x):
        y_[...] = x_[...]
        y_[...][...] = x_*numpy.exp(x)


    @statement(ain=['x'], aout=['y'])
    def sum(engine, x, y):
        y[...] = engine.pm.comm.allreduce(x[...]).sum(dtype='f8')

    @sum.defvjp
    def _(engine, _y, _x, x):
        _x[...] = numpy.ones_like(x) * _y

    @sum.defjvp
    def _(engine, y_, x_):
        y_[...] = engine.pm.comm.allreduce(x_).sum(dtype='f8')


    @statement(aout=['v'], ain=['v', 'f'])
    def kick(engine, v, f, kick_factor):
        v[...] += f * kick_factor

    @kick.defvjp
    def _(engine, _f, _v, kick_factor):
        _f[...] = _v * kick_factor

    @kick.defjvp
    def _(engine, f_, v_, kick_factor):
        v_[...] += f_ * kick_factor

    @statement(aout=['x'], ain=['x', 'v'])
    def drift(engine, x, v, drift_factor):
        x[...] += v * drift_factor

    @drift.defvjp
    def _(engine, _x, _v, drift_factor):
        _v[...] = _x * drift_factor

    @drift.defjvp
    def _(engine, x_, v_, v, drift_factor):
        x_[...] += v_ * drift_factor



##    @programme(ain=['final'], aout=['model', 'decic'])
##    def apply_halomodel(engine, model, final, posdata, mdata, R1, R2):
##        """ Apply a trained neural network to predict
##
##            This will return a vector of predictions based on the input features and
##        and the weights and intercepts of the trained Neural Network
##        """
##        code = CodeSegment(engine)
##
##        print('halomodel intiate')
##        pmx, psx, pcoef, pintercept = posdata
##        mmx, msx, mmy, msy, mcoef, mintercept = mdata
##
##        ##Generate differet smoothed fields
## 
##        code.r2c(real='final', complex='d_k')
##        code.de_cic(deconvolved='decic', d_k='d_k')
##        code.r2c(real='decic', complex='d_k')
##        code.fingauss_smoothing(smoothed='R1', R=R1, d_k='d_k')
##        code.fingauss_smoothing(smoothed='R2', R=R2, d_k='d_k')
##        #code.multiply(x1='R2', x2=Literal(-1), y='negR2')
##        #code.add(x1='R1', x2='negR2', y='R12')
##
##        ##Create feature array of 27neighbor field for all 
##        names = ['final', 'R1', 'R2']
##        N = len(engine.q)
##        Nf, Nnb = len(names), 27
##        Ny = Nf*Nnb
##        code.assign(x=Literal(numpy.zeros((N, Ny))), y='pfeature')
##        code.assign(x=Literal(numpy.zeros((N, Nf))), y='mfeature')
##        grid = engine.pm.generate_uniform_particle_grid(shift=0)
##        layout = engine.pm.decompose(grid)
##
##        for i in range(Nf):
##            #p   
##            code.find_neighbours(field=names[i], features='ptmp')
##
##            #normalize feature 
##            code.add(x1='ptmp', x2=Literal(-1*pmx[i*Nnb:(i+1)*Nnb]), y='ptmp1')
##            code.multiply(x1='ptmp1', x2=Literal(psx[i*Nnb:(i+1)*Nnb]**-1), y='ptmp2')
##            code.assign_chunk(attribute='pfeature', value='ptmp2', start=i*Nnb, end=Nnb*(i+1))
##
##            #m 
##            code.readout(x=Literal(grid), mesh=names[i], value='mtmp', layout=Literal(layout), resampler='nearest')
##
##            #normalize feature
##            code.add(x1='mtmp', x2=Literal(-1*mmx[i]), y='mtmp1')
##            code.multiply(x1='mtmp1', x2=Literal(psx[i]**-1), y='mtmp2')
##            code.assign_component(attribute='mfeature', value='mtmp2', dim=i)
##
##        code.apply_nets(predict='ppredict', features='pfeature', coeff=pcoef, intercept=pintercept, Nd=N, prob=True, classify=True)
##        code.apply_nets(predict='mpredict', features='mfeature', coeff=mcoef, intercept=mintercept, Nd=N, prob=False)
##
##        #renormalize mass 
##        code.multiply(x1='mpredict', x2=Literal(msy), y='mpredict')
##        code.add(x1='mpredict', x2=Literal(mmy), y='mpredict')
##        code.reshape_scalar(x='ppredict', y='ppredict')
##        code.reshape_scalar(x='mpredict', y='mpredict')
##
##        #paint 
##        code.paint(x=Literal(grid), mesh='posmesh', layout=Literal(layout), mass='ppredict')
##        code.paint(x=Literal(grid), mesh='massmesh', layout=Literal(layout), mass='mpredict')
##        code.multiply(x1='posmesh', x2='massmesh', y='premodel')
##
##        #Smooth
##        #code.assign(x='premodel', y='model')                                                                                                                  
##        code.r2c(real='premodel', complex='d_k')
##        code.fingauss_smoothing(smoothed='model', R=4, d_k='d_k')
##
##        return code
##



##    @programme(ain=['features'], aout=['predict'])
##    def apply_nets_old(engine, predict, features, coeff, intercept, Nd, active = 'relu', prob = False, classify=False, t=0, w=1):
##        """ Apply a trained neural network to predict
##
##            This will return a vector of predictions based on the input features and
##        and the weights and intercepts of the trained Neural Network
##        """
##        code = CodeSegment(engine)
##        nl = len(coeff)
##        code.assign(x=Literal(numpy.zeros((Nd, coeff[0].shape[1]))), y='tmp')
##        code.assign(x='features', y='inp')
##        #print('start loop')
##
##        for i in range(nl):
##            code.matrix_cmul(W=coeff[i], x='inp', y='tmp')
##            code.add(x1='tmp', x2=Literal(intercept[i]), y='tmp')
##            if i < nl-1:
##                if active == 'relu':
##                    code.relu(x='tmp', y='tmp')
##                elif active == 'linear':
##                    code.identity(x='tmp', y='tmp')
##                code.assign(x='tmp', y='inp')
##                if i == nl-2:
##                    code.assign(x=Literal(numpy.zeros((Nd))), y='tmp')
##                else:
##                    code.assign(x=Literal(numpy.zeros((Nd, coeff[i+1].shape[1]))), y='tmp')
##            else: #ie if its the final output, check if we are calculating the probability
##                if prob:
##                    code.logistic(x='tmp', y='tmp', t=t, w=w)
##                    if classify:
##                        code.threshold(x='tmp', y='tmp', t=0.5)
##                    #print('Logistic')
##        code.assign(x='tmp', y='predict')
##        return code
