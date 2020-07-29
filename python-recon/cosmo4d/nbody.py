from . import base
from .engine import ParticleMesh, FastPMEngine, CodeSegment, Literal
from fastpm.background import PerturbationGrowth

class NBodyModel(base.DynamicModel):
    def __init__(self, cosmo, pm, B, steps, order=2):
        pt = PerturbationGrowth(cosmo)
        self.cosmo = cosmo
        self.pt = pt
        self.pm = pm
        self.Nmesh = pm.Nmesh
        self.BoxSize = pm.BoxSize
        self.engine = FastPMEngine(self.pm, B=B)
        self.order = order

        self.steps = steps

    def get_code(self):
        engine = self.engine
        code = CodeSegment(engine)
        code.r2c(real='parameters', complex='dlinear_k')

        if len(self.steps) > 0:
            code.solve_fastpm(pt=self.pt, dlinear_k='dlinear_k', asteps=self.steps, s='s', order=self.order)
            code.get_x(s='s', x='x')
            code.paint_simple(x='x', density='final')
        else:
            code.c2r(complex='dlinear_k', real='final')
            code.assign(x=Literal(engine.q), y='x')

        code.assign(x='x', y='X') # X is the position
        code.multiply(x1='v', x2=Literal(1.0), y='V') # unit is wrong.
        return code


class LPTModel(base.DynamicModel):
    def __init__(self, cosmo, pm, B, steps):
        pt = PerturbationGrowth(cosmo)
        self.cosmo = cosmo
        self.pt = pt
        self.pm = pm
        self.Nmesh = pm.Nmesh
        self.BoxSize = pm.BoxSize
        self.engine = FastPMEngine(self.pm, B=B)

        self.steps = steps

    def get_code(self):
        engine = self.engine
        code = CodeSegment(engine)
        code.r2c(real='parameters', complex='dlinear_k')

        if len(self.steps) > 0:
            code.solve_lpt(pt=self.pt, dlinear_k='dlinear_k', aend=self.steps[-1], s='s')
            code.get_x(s='s', x='x')
            code.paint_simple(x='x', density='final')
        else:
            code.c2r(complex='dlinear_k', real='final')
            code.assign(x=Literal(engine.q), y='x')

        code.assign(x='x', y='X') # X is the position
        code.multiply(x1='v', x2=Literal(1.0), y='V') # unit is wrong.
        return code


class ZAModel(base.DynamicModel):
    def __init__(self, cosmo, pm, B, steps):
        pt = PerturbationGrowth(cosmo)
        self.cosmo = cosmo
        self.pt = pt
        self.pm = pm
        self.Nmesh = pm.Nmesh
        self.BoxSize = pm.BoxSize
        self.engine = FastPMEngine(self.pm, B=B)

        self.steps = steps

    def get_code(self):
        engine = self.engine
        code = CodeSegment(engine)
        code.r2c(real='parameters', complex='dlinear_k')

        if len(self.steps) > 0:
            code.solve_za(pt=self.pt, dlinear_k='dlinear_k', aend=self.steps[-1], s='s')
            code.get_x(s='s', x='x')
            code.paint_simple(x='x', density='final')
        else:
            code.c2r(complex='dlinear_k', real='final')
            code.assign(x=Literal(engine.q), y='x')

        code.assign(x='x', y='X') # X is the position
        code.multiply(x1='v', x2=Literal(1.0), y='V') # unit is wrong.
        return code

    
class NBodyLinModel(base.DynamicModel):
    def __init__(self, cosmo, pm, B, steps):
        pt = PerturbationGrowth(cosmo)
        self.pt = pt
        self.pm = pm
        self.Nmesh = pm.Nmesh
        self.BoxSize = pm.BoxSize
        self.engine = FastPMEngine(self.pm, B=B)

        self.steps = steps

    def get_code(self):
        engine = self.engine
        code = CodeSegment(engine)
        code.r2c(real='parameters', complex='dlinear_k')
        code.c2r(complex='dlinear_k', real='final')
        
        code.assign(x=Literal(engine.q), y='x')

        code.assign(x='x', y='X') # X is the position
        code.multiply(x1='v', x2=Literal(1.0), y='V') # unit is wrong.
        return code



class FFTModel(base.DynamicModel):
    def __init__(self, pm):
        self.pm = pm
        self.Nmesh = pm.Nmesh
        self.BoxSize = pm.BoxSize
        self.engine = FastPMEngine(self.pm, B=1)


    def get_code(self):
        engine = self.engine
        code = CodeSegment(engine)
        code.r2c(real='parameters', complex='dlinear_k')
        code.c2r(complex='dlinear_k', real='final')
        return code


