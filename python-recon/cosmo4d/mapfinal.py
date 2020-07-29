from . import base
from .engine import Literal
from .iotools import save_map, load_map
from nbodykit.lab import FieldMesh

class Observable(base.Observable):
    def __init__(self, mapp, d, s):
        self.mapp = mapp
        self.s = s
        self.d = d

    def save(self, path):
        save_map(self.mapp, path, 'mapp')
        save_map(self.s, path, 's')
        save_map(self.d, path, 'd')

    @classmethod
    def load(kls, path):
        return Observable(load_map(path, 'mapp'),
                          load_map(path, 'd'),
                          load_map(path, 's'))
    def downsample(self, pm):
        return Observable(
                pm.downsample(self.mapp, resampler='nearest', keep_mean=True),
                pm.downsample(self.d, resampler='nearest', keep_mean=True),
                pm.downsample(self.s, resampler='nearest', keep_mean=True),
        )




class MockModel(base.MockModel):
    def __init__(self, dynamic_model):
        self.dynamic_model = dynamic_model
        self.pm = dynamic_model.pm
        self.engine = dynamic_model.engine

    def get_code(self):
        code = base.MockModel.get_code(self)
        code.assign(x='final', y='model')
        return code

    def make_observable(self, initial):
        code = self.get_code()
        model, final = code.compute(['model', 'final'], init={'parameters':initial})
        return Observable(mapp=model, s=initial, d=final)



class DataModel(base.MockModel):
    def __init__(self, dynamic_model):
        self.dynamic_model = dynamic_model
        self.pm = dynamic_model.pm
        self.engine = dynamic_model.engine

    def get_code(self):
        code = base.MockModel.get_code(self)
        code.assign(x='final', y='model')
        return code

    def make_observable(self, initial):
        code = self.get_code()
        model, final = code.compute(['model', 'final'], init={'parameters':initial})
        return Observable(mapp=model, s=initial, d=final)




class NoiseModel(base.NoiseModel):
    def __init__(self, pm, mask2d, power, seed):
        self.pm = pm
        self.pm2d = self.pm.resize([self.pm.Nmesh[0], self.pm.Nmesh[1], 1])
        if mask2d is None:
            mask2d = self.pm2d.create(mode='real')
            mask2d[...] = 1.0

        self.mask2d = mask2d
        self.power = power
        self.seed = seed
        self.var= power / (self.pm.BoxSize / self.pm.Nmesh).prod()
        self.ivar2d = mask2d * self.var ** -1

    def downsample(self, pm):
        d = NoiseModel(pm, None, self.power, self.seed)
        d.mask2d = d.pm2d.downsample(self.mask2d)
        return d

    def add_noise(self, obs):
        pm = obs.mapp.pm

        if self.seed is None:
            n = pm.create(mode='real')
            n[...] = 0
        else:
            n = pm.generate_whitenoise(mode='complex', seed=self.seed)
            n = n.apply(lambda k, v : (self.power / pm.BoxSize.prod()) ** 0.5 * v, out=Ellipsis).c2r(out=Ellipsis)
            print('Noise Variance check', (n ** 2).csum() / n.Nmesh.prod(), self.var)
        return Observable(mapp=obs.mapp + n, s=obs.s, d=obs.d)



class Objective(base.Objective):
    def __init__(self, mock_model, noise_model, data, prior_ps):
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine

    def get_code(self):
        pm = self.mock_model.pm

        code = base.Objective.get_code(self)

        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        code.to_scalar(x='residual', y='chi2')
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code

    def evaluate(self, model, data):
        from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower

        xm = FFTPower(first=FieldMesh(model.mapp/model.mapp.cmean()), second=FieldMesh(data.mapp/data.mapp.cmean()), mode='1d')
        xd = FFTPower(first=FieldMesh(model.d), second=FieldMesh(data.d), mode='1d')
        xs = FFTPower(first=FieldMesh(model.s), second=FieldMesh(data.s), mode='1d')

        pm1 = FFTPower(first=FieldMesh(model.mapp/model.mapp.cmean()), mode='1d')
        pd1 = FFTPower(first=FieldMesh(model.d), mode='1d')
        ps1 = FFTPower(first=FieldMesh(model.s), mode='1d')

        pm2 = FFTPower(first=FieldMesh(data.mapp/data.mapp.cmean()), mode='1d')
        pd2 = FFTPower(first=FieldMesh(data.d), mode='1d')
        ps2 = FFTPower(first=FieldMesh(data.s), mode='1d')

        data_preview = dict(s=[], d=[], mapp=[])
        model_preview = dict(s=[], d=[], mapp=[])

        for axes in [[1, 2], [0, 2], [0, 1]]:
            data_preview['d'].append(data.d.preview(axes=axes))
            data_preview['s'].append(data.s.preview(axes=axes))
            data_preview['mapp'].append(data.mapp.preview(axes=axes))
            model_preview['d'].append(model.d.preview(axes=axes))
            model_preview['s'].append(model.s.preview(axes=axes))
            model_preview['mapp'].append(model.mapp.preview(axes=axes))

        #data_preview['mapp'] = data.mapp.preview(axes=(0, 1))
        #model_preview['mapp'] = model.mapp.preview(axes=(0, 1))

        return xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power, data_preview, model_preview

    def save_report(self, report, filename):
        xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2, data_preview, model_preview = report

        km = xm['k']
        ks = xs['k']
        kd = xd['k']

        xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
        xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
        xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

        tm = (pm1['power'] / pm2['power']) ** 0.5
        ts = (ps1['power'] / ps2['power']) ** 0.5
        td = (pd1['power'] / pd2['power']) ** 0.5

        from cosmo4d.iotools import create_figure
        fig, gs = create_figure((12, 9), (4, 6))
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(data_preview['s'][i])
            ax.set_title("s data")

        for i in range(3):
            ax = fig.add_subplot(gs[0, i + 3])
            ax.imshow(data_preview['d'][i])
            ax.set_title("d data")

        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(model_preview['s'][i])
            ax.set_title("s model")

        for i in range(3):
            ax = fig.add_subplot(gs[1, i + 3])
            ax.imshow(model_preview['d'][i])
            ax.set_title("d model")

        for i in range(3):
            ax = fig.add_subplot(gs[2, i + 3])
            ax.imshow(data_preview['mapp'][i])
            ax.set_title("map data")

        for i in range(3):
            ax = fig.add_subplot(gs[3, i + 3])
            ax.imshow(model_preview['mapp'][i])
            ax.set_title("map model")

        ax = fig.add_subplot(gs[2, 0])
        ax.plot(ks, xs, label='intial')
        ax.plot(kd, xd, label='final')
        ax.plot(km, xm, label='map')
        ax.legend()
        ax.set_xscale('log')
        ax.set_title('Cross coeff')

        ax = fig.add_subplot(gs[2, 1])
        ax.plot(ks, ts, label='intial')
        ax.plot(kd, td, label='final')
        ax.plot(km, tm, label='map')
        ax.legend()
        ax.set_xscale('log')
        ax.set_title('Transfer func')

        ax = fig.add_subplot(gs[2, 2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.05, 0.9, 's=linear')
        ax.text(0.05, 0.7, 'd=non-linear')
        ax.text(0.05, 0.5, 'map=halos (sm)')
        ax.text(0.05, 0.3, 'model=FastPM+NN')
        ax.text(0.05, 0.1, 'data=FastPM+NN')

        ax = fig.add_subplot(gs[3, 0])
        ax.plot(ks, ps1['power'], label='model')
        ax.plot(ks, ps2['power'], label='data')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("initial")
        ax.legend()

        ax = fig.add_subplot(gs[3, 1])
        ax.plot(kd, pd1['power'], label='model')
        ax.plot(kd, pd2['power'], label='data')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("final")
        ax.legend()

        ax = fig.add_subplot(gs[3, 2])
        ax.plot(km, pm1['power'], label='model')
        ax.plot(km, pm2['power'], label='data')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("map")
        ax.legend()


        fig.tight_layout()
        if self.pm.comm.rank == 0:
            fig.savefig(filename)

class SmoothedObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml

    def get_code(self):
        import numpy
        pm = self.mock_model.pm

        code = self.mock_model.get_code()

        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        code.c2r(real='residual', complex='C')
        code.to_scalar(x='residual', y='chi2')
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code

