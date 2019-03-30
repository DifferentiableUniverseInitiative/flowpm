import numpy as np
import numpy, os, sys
from nbodykit.lab import BigFileMesh, BigFileCatalog, FieldMesh, ArrayCatalog, KDDensity
#from nbodykit.source.mesh.field import FieldMesh

from pmesh.pm import ParticleMesh

#from nbodykit.cosmology import Cosmology, EHPower, Planck15
from background import *
from fpmfuncs import *
from pmconfig import Config
####


def lpt1(dlin_k, q, resampler='cic'):
    """ Run first order LPT on linear density field, returns displacements of particles
        reading out at q. The result has the same dtype as q.
    """
    basepm = dlin_k.pm

    ndim = len(basepm.Nmesh)
    delta_k = basepm.create('complex')

    layout = basepm.decompose(q)
    local_q = layout.exchange(q)

    source = numpy.zeros((len(q), ndim), dtype=q.dtype)
    for d in range(len(basepm.Nmesh)):
        disp = dlin_k.apply(laplace) \
                    .apply(gradient(d), out=Ellipsis) \
                    .c2r(out=Ellipsis)
        local_disp = disp.readout(local_q, resampler=resampler)
        source[..., d] = layout.gather(local_disp)
    return source


def lpt2source(dlin_k):
    """ Generate the second order LPT source term.  """
    source = dlin_k.pm.create('real')
    source[...] = 0
    if dlin_k.ndim != 3: # only for 3d
        return source.r2c(out=Ellipsis)

    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []

    # diagnoal terms
    for d in range(dlin_k.ndim):
        phi_ii_d = dlin_k.apply(laplace) \
                     .apply(gradient(d), out=Ellipsis) \
                     .apply(gradient(d), out=Ellipsis) \
                     .c2r(out=Ellipsis)
        phi_ii.append(phi_ii_d)

    for d in range(3):
        source[...] += phi_ii[D1[d]].value * phi_ii[D2[d]].value

    # free memory
    phi_ii = []

    phi_ij = []
    # off-diag terms
    for d in range(dlin_k.ndim):
        phi_ij_d = dlin_k.apply(laplace) \
                 .apply(gradient(D1[d]), out=Ellipsis) \
                 .apply(gradient(D2[d]), out=Ellipsis) \
                 .c2r(out=Ellipsis)

        source[...] -= phi_ij_d[...] ** 2

    # this ensures x = x0 + dx1(t) + d2(t) for 2LPT

    source[...] *= 3.0 / 7
    return source.r2c(out=Ellipsis)


def lptz0( lineark, Q, a=1, order=2):
    """ This computes the 'force' from LPT as well. """

    DX1 = 1 * lpt1(lineark, Q)

    if order == 2:
        DX2 = 1 * lpt1(lpt2source(lineark), Q)
    else:
        DX2 = 0
    return DX1 + DX2



###############

class StateVector(object):
    def __init__(self, solver, Q):
        self.solver = solver
        self.pm = solver.pm
        self.Q = Q
        self.csize = solver.pm.comm.allreduce(len(self.Q))
        self.dtype = self.Q.dtype
        self.cosmology = solver.cosmology

        self.H0 = 100. # in km/s / Mpc/h units
        # G * (mass of a particle)
        self.GM0 = self.H0 ** 2 / ( 4 * numpy.pi ) * 1.5 * self.cosmology.Om0 * self.pm.BoxSize.prod() / self.csize

        self.S = numpy.zeros_like(self.Q)
        self.P = numpy.zeros_like(self.Q)
        self.F = numpy.zeros_like(self.Q)
        self.RHO = numpy.zeros_like(self.Q[..., 0])
        self.a = dict(S=None, P=None, F=None)

    def copy(self):
        obj = object.__new__(type(self))
        od = obj.__dict__
        od.update(self.__dict__)
        obj.S = self.S.copy()
        obj.P = self.P.copy()
        obj.F = self.F.copy()
        obj.RHO = self.RHO.copy()
        return obj

    @property
    def X(self):
        return self.S + self.Q

    @property
    def V(self):
        a = self.a['P']
        return self.P * (self.H0 / a)

########
class Solver(object):
    def __init__(self, pm, cosmology, B=1, a_linear=1.0):
        """
            a_linear : float
                scaling factor at the time of the linear field.
                The growth function will be calibrated such that at a_linear D1 == 0.

        """
        if not isinstance(cosmology, Cosmology):
            raise TypeError("only nbodykit.cosmology object is supported")

        fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=pm.dtype, comm=pm.comm, resampler=pm.resampler)
        self.pm = pm
        self.fpm = fpm
        self.cosmology = cosmology
        self.a_linear = a_linear

    # override nbodystep in subclasses
    @property
    def nbodystep(self):
        return FastPMStep(self)

    def whitenoise(self, seed, unitary=False):
        return self.pm.generate_whitenoise(seed, type='complex', unitary=unitary)

    def linear(self, whitenoise, Pk):
        return whitenoise.apply(lambda k, v:
                        Pk(sum(ki ** 2 for ki in k)**0.5) ** 0.5 * v / v.BoxSize.prod() ** 0.5)

    def lpt(self, linear, Q, a, order=2):
        """ This computes the 'force' from LPT as well. """
        assert order in (1, 2)

#         from .force.lpt import lpt1, lpt2source

        state = StateVector(self, Q)
        pt = PerturbationGrowth(self.cosmology, a=[a], a_normalize=self.a_linear)
        DX1 = pt.D1(a) * lpt1(linear, Q)

        V1 = a ** 2 * pt.f1(a) * pt.E(a) * DX1
        if order == 2:
            DX2 = pt.D2(a) * lpt1(lpt2source(linear), Q)
            V2 = a ** 2 * pt.f2(a) * pt.E(a) * DX2
            state.S[...] = DX1 + DX2
            state.P[...] = V1 + V2
            state.F[...] = a ** 2 * pt.E(a) * (pt.gf(a) / pt.D1(a) * DX1 + pt.gf2(a) / pt.D2(a) * DX2)
        else:
            state.S[...] = DX1
            state.P[...] = V1
            state.F[...] = a ** 2 * pt.E(a) * (pt.gf(a) / pt.D1(a) * DX1)

        state.a['S'] = a
        state.a['P'] = a

        return state

    def nbody(self, state, stepping, monitor=None):
        step = self.nbodystep
        for action, ai, ac, af in stepping:
            step.run(action, ai, ac, af, state, monitor)

        return state

    


class FastPMStep(object):
    def __init__(self, solver):
        self.cosmology = solver.cosmology
        self.pm = solver.fpm
        self.solver = solver

    def run(self, action, ai, ac, af, state, monitor):
        actions = dict(K=self.Kick, D=self.Drift, F=self.Force)

        event = actions[action](state, ai, ac, af)
        if monitor is not None:
            monitor(action, ai, ac, af, state, event)

    def Kick(self, state, ai, ac, af):
        assert ac == state.a['F']
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac)
        state.P[...] = state.P[...] + fac * state.F[...]
        state.a['P'] = af

    def Drift(self, state, ai, ac, af):
        assert ac == state.a['P']
        pt = PerturbationGrowth(self.cosmology, a=[ai, ac, af], a_normalize=self.solver.a_linear)
        fac = 1 / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac)
        state.S[...] = state.S[...] + fac * state.P[...]
        state.a['S'] = af

    def prepare_force(self, state, smoothing):
        nbar = 1.0 * state.csize / self.pm.Nmesh.prod()
        X = state.X
        layout = self.pm.decompose(X, smoothing)
        X1 = layout.exchange(X)
        rho = self.pm.paint(X1)
        rho /= nbar # 1 + delta
        return layout, X1, rho

    def Force(self, state, ai, ac, af):

        assert ac == state.a['S']
        # use the default PM support
        layout, X1, rho = self.prepare_force(state, smoothing=None)
        state.RHO[...] = layout.gather(rho.readout(X1))
        delta_k = rho.r2c(out=Ellipsis)
        state.F[...] = layout.gather(longrange(X1, delta_k, split=0, factor=1.5 * self.cosmology.Om0))
        state.a['F'] = af
        return dict(delta_k=delta_k)



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



def fastpm(lptinit=False):
    config = Config()

    solver = Solver(config.pm, cosmology=config['cosmology'], B=config['pm_nc_factor'])
    whitenoise = solver.whitenoise(seed=config['seed'], unitary=config['unitary'])
    dlin = solver.linear(whitenoise, Pk=lambda k : config['powerspectrum'](k))

    Q = config.pm.generate_uniform_particle_grid(shift=config['shift'])

    state = solver.lpt(dlin, Q=Q, a=config['stages'][0], order=2)
    if lptinit: return state

    def monitor(action, ai, ac, af, state, event):
        if config.pm.comm.rank == 0:
            print('Step %s %06.4f - (%06.4f) -> %06.4f' %( action, ai, ac, af),
                  'S %(S)06.4f P %(P)06.4f F %(F)06.4f' % (state.a))


    solver.nbody(state, stepping=leapfrog(config['stages']), monitor=monitor)
    return state




if __name__=="__main__":

    sys.path.append('../../utils')
    import tools

    bs, nc = 400, 128
    z = 0
    nsteps = 5
    seed = 100
    seeds = np.arange(100, 1100, 100)

    for seed in seeds:
        print('\nDo for seed = %d\n'%seed)
        #Setup
        conf = Config(bs=bs, nc=nc, seed=seed)
        pm = conf.pm
        assert conf['stages'].size == nsteps

        grid = pm.generate_uniform_particle_grid(shift=0).astype(np.float32)
        kvec = tools.fftk((nc, nc, nc), bs, dtype=np.float32, symmetric=False)
        solver = Solver(pm, conf['cosmology'])
        conf['kvec'] = kvec
        conf['grid'] = grid

        #PM
        #whitec = pm.generate_whitenoise(seed, mode='complex', unitary=False)
        #lineark = whitec.apply(lambda k, v:Planck15.get_pklin(sum(ki ** 2 for ki in k)**0.5, 0) ** 0.5 * v / v.BoxSize.prod() ** 0.5)
        #linear = lineark.c2r()
        linear = BigFileMesh('/project/projectdirs/astro250/chmodi/cosmo4d/data/z00/L0400_N0128_S0100_05step/mesh/', 's').paint()
        lineark = linear.r2c()
        state = solver.lpt(lineark, grid, conf['stages'][0], order=2)
        solver.nbody(state, leapfrog(conf['stages']))
        final = pm.paint(state.X)
        if pm.comm.rank == 0:
            print('X, V computed')
        cat = ArrayCatalog({'Position': state.X, 'Velocity' : state.V}, BoxSize=pm.BoxSize, Nmesh=pm.Nmesh)
        kdd = KDDensity(cat).density
        cat['KDDensity'] = kdd

        #Save
        path = '/project/projectdirs/astro250/chmodi/cosmo4d/data/'
        ofolder = path + 'z%02d/L%04d_N%04d_S%04d_%02dstep_fpm/'%(z*10, bs, nc, seed, nsteps)
        FieldMesh(linear).save(ofolder+'mesh', dataset='s', mode='real')
        FieldMesh(final).save(ofolder+'mesh', dataset='d', mode='real')
        cat.save(ofolder + 'dynamic/1', ('Position', 'Velocity', 'KDDensity'))

