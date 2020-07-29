from abopt.abopt2 import Problem

class Observable(object):
    def __init__(self, *args, **kwargs):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(kls, path):
        pass

class Differentiable(object):
    def __init__(self): pass

    def get_code(self):
        raise NotImplementedError


class DynamicModel(Differentiable):
    def __init__(self, *args, **kwargs):
        pass

    def get_code(self):
        pass

    def compute(self, initial):
        final = None
        return final

class MockModel(object):
    def __init__(self, dynamic_model, *args, **kwargs):
        self.dynamic_model = dynamic_model

    def make_observable(self, final):
        obs = Observable()
        return obs

    def get_code(self):
        return self.dynamic_model.get_code().copy()

class NoiseModel(object):
    def __init__(self, *args, **kwargs):
        pass

    def add_noise(self, obs):
        return obs # add noise

class Objective(Differentiable):
    def __init__(self, mock_model, noise_model, data):
        assert isinstance(data, Observable)
        self.data = data
        self.mock_model = mock_model
        self.engine = mock_model.engine

    def get_code(self):
        return self.mock_model.get_code().copy() # subclass add more to compute objecgtive

    def get_problem(self, precond=None, atol=1e-5, rtol=1e-5):
        """ a problem object for abopt's optimizers """
        code = self.get_code()

        def compute(s, vout='objective'):
            if self.engine.pm.comm.rank == 0:
                print('Compute objective')
            return code.compute(vout, init={'parameters' : s})

        def gradient(s, var='_parameters'):
            if self.engine.pm.comm.rank == 0:
                print('Compute gradient')
            r, tape = code.compute('objective', init={'parameters': s}, return_tape=True)
            gradient = tape.get_vjp()
            return gradient.compute(var, init={'_objective' : 1.0})

        #FIXME: this is wrong if the objective is not only prior + chi2
        def hessian_dot(s, v):
            """ This creates the gauss-newton hessian """

            r, tape_x = code.compute('residual', init={'parameters': s}, return_tape=True)
            jvp_x = tape_x.get_jvp()
            vjp_x = tape_x.get_vjp()

            def hessian_dot(jvp, vjp, v, var):
                JTv = vjp.compute('_parameters', init={'_' + var: v})
                return jvp.compute(var + '_', init={'parameters_': JTv})

            r, tape_p = code.compute('whitenoise', init={'parameters': s}, return_tape=True)
            jvp_p = tape_p.get_jvp()
            vjp_p = tape_p.get_vjp()

            return (hessian_dot(jvp_x, vjp_x, v, 'residual')
                  + hessian_dot(jvp_p, vjp_p, v, 'whitenoise'))

        problem = Problem(
                compute,
                gradient,
                vs=self.engine.vs,
                hessian_vector_product=hessian_dot,
                precond=precond, atol=atol, rtol=rtol
                )

        problem.compute = compute
        return problem


