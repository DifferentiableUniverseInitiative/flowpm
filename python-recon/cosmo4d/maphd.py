## Main model- use NN to generate halo mass field and optimize using 
## loss function of (M_R - M_data/ M_0) form.
## Noise and offset can be 3d 

import numpy
from . import base
from .engine import Literal
from .iotools import save_map, load_map
from nbodykit.lab import FieldMesh
import re, json, warnings

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
    def __init__(self, dynamic_model, ppath, mpath, pwidth = None):
        self.dynamic_model = dynamic_model
        self.pm = dynamic_model.pm
        self.engine = dynamic_model.engine
        self.ppath = ppath
        self.mpath = mpath
        #self.R1p, self.R2p = R1, R2
        self.pwidth = pwidth

        self._setup_pNN()
        self._setup_mNN()


    def _setup_pNN(self):

        ppath = self.ppath
        with open(ppath + '/pinfo.json') as fp: pdict = json.load(fp)
        self.R1p, self.R2p = pdict['R1'], pdict['R2']

        self.pftname = pdict['pftname']
        acts = pdict['activations']
        self.pacts = ['sigmoid' if 'sigmoid' in s else s for s in acts]
        if self.pwidth is None:
            self.pwidth = pdict['width']

        self.pwts, self.pbias = [], []

        for s in [0, 2, 4]:
            try:
                self.pwts.append(numpy.load(ppath + 'w%d.npy'%s))
                self.pbias.append(numpy.load(ppath + 'b%d.npy'%s))
            except:
                pass
        self.pmx = numpy.load(ppath + 'mx.npy')
        self.psx = numpy.load(ppath + 'sx.npy')
        self.parch = tuple(zip(self.pwts, self.pbias, self.pacts))
        if self.pm.comm.rank == 0:
            print('Position Netowrk built from path \n %s\n'%self.ppath)
            print('Network architecture for position')
            for i, ar in enumerate(self.parch):
                print('layer %d has shape %s, followed by activation %s '%(i, str(ar[0].shape), ar[2]))
            print('and width of sigmoid = %d \n'%self.pwidth)
        if len(self.pwts) != len(self.pacts) :
            print('Inconsistent Network, length of weights not the same as activations')
            print(len(self.pwts), len(self.pacts))
            import sys
            sys.exit()



    def _setup_mNN(self):

        mpath = self.mpath
        with open(mpath + '/minfo.json') as fp: pdict = json.load(fp)

        self.mftname = pdict['mftname']
        self.macts = pdict['activations']
        self.mwidth = 0
            
        self.mwts, self.mbias = [], []

        for s in [0, 2, 4]:
            try:
                self.mwts.append(numpy.load(mpath + 'w%d.npy'%s))
                self.mbias.append(numpy.load(mpath + 'b%d.npy'%s))
            except:
                pass
        self.mmx = numpy.load(mpath + 'mx.npy')
        self.msx = numpy.load(mpath + 'sx.npy')
        self.mmy = numpy.load(mpath + 'my.npy')
        self.msy = numpy.load(mpath + 'sy.npy')

        self.march = tuple(zip(self.mwts, self.mbias, self.macts))
        if self.pm.comm.rank == 0:
            print('Mass Netowrk built from path \n %s\n'%self.mpath)
            print('Network architecture for mass')
            for i, ar in enumerate(self.march):
                print('layer %d has shape %s, followed by activation %s '%(i, str(ar[0].shape), ar[2]))
            print('\n')
        if len(self.mwts) != len(self.macts) :
            print('Inconsistent Network, length of weights not the same as activations')
            print(len(self.mwts), len(self.macts))
            import sys
            sys.exit()

    def get_code(self):
        code = self.dynamic_model.get_code()
        ##Generate differet smoothed fields
        code.r2c(real='final', complex='d_k')
        code.de_cic(deconvolved='decic', d_k='d_k')
        #subtract mean
        code.add(x1='decic', x2=Literal(-1.), y='decic')
        #
        code.r2c(real='decic', complex='d_k')
        code.fingauss_smoothing(smoothed='R1', R=self.R1p, d_k='d_k')
        code.fingauss_smoothing(smoothed='R2', R=self.R2p, d_k='d_k')
        code.multiply(x1='R2', x2=Literal(-1), y='negR2')
        code.add(x1='R1', x2='negR2', y='R12')

        ##Create feature array of 27neighbor field for all
        #names = self.mftname
        N = len(self.engine.q)
        Npf = len(self.pftname)
        Nm = len(self.mftname)
        if self.pwts[0].shape[0] % 27:
            Nnb = 1
        else:
            Nnb = 27
        Np = Npf*Nnb
        code.assign(x=Literal(numpy.zeros((N, Np))), y='pfeature')
        code.assign(x=Literal(numpy.zeros((N, Nm))), y='mfeature')
        grid = self.pm.generate_uniform_particle_grid(shift=0)
        layout = self.engine.pm.decompose(grid)

        #pos
        for i in range(Npf):
            #p
            if Nnb == 27:
                code.find_neighbours(field=self.pftname[i], features='ptmp')
            else:
                code.readout(x=Literal(grid), mesh=self.pftname[i], value='ptmp', layout=Literal(layout), resampler='nearest')                
            #normalize feature
            code.add(x1='ptmp', x2=Literal(-1*self.pmx[i*Nnb:(i+1)*Nnb]), y='ptmp1')
            code.multiply(x1='ptmp1', x2=Literal(self.psx[i*Nnb:(i+1)*Nnb]**-1), y='ptmp2')
            if Nnb == 27:
                code.assign_chunk(attribute='pfeature', value='ptmp2', start=i*Nnb, end=Nnb*(i+1))
            else:
                code.assign_component(attribute='pfeature', value='ptmp2', dim=i)
        #mass
        for i in range(Nm):
            #m
            code.readout(x=Literal(grid), mesh=self.mftname[i], value='mtmp', layout=Literal(layout), resampler='nearest')
            #normalize feature
            code.add(x1='mtmp', x2=Literal(-1*self.mmx[i]), y='mtmp1')
            code.multiply(x1='mtmp1', x2=Literal(self.msx[i]**-1), y='mtmp2')
            code.assign_component(attribute='mfeature', value='mtmp2', dim=i)
            
        code.apply_nets(predict='ppredict', features='pfeature', arch=self.parch, Nd=N, t=0, w=self.pwidth)
        code.apply_nets(predict='mpredict', features='mfeature', arch=self.march, Nd=N, t=0, w=self.mwidth)

        #renormalize mass
        code.multiply(x1='mpredict', x2=Literal(self.msy), y='mpredict')
        code.add(x1='mpredict', x2=Literal(self.mmy), y='mpredict')
        code.reshape_scalar(x='ppredict', y='ppredict')
        code.reshape_scalar(x='mpredict', y='mpredict')
        #paint
        code.paint(x=Literal(grid), mesh='posmesh', layout=Literal(layout), mass='ppredict')
        code.paint(x=Literal(grid), mesh='massmesh', layout=Literal(layout), mass='mpredict')
        code.multiply(x1='posmesh', x2='massmesh', y='premodel')
        #Smooth
        code.assign(x='premodel', y='model')
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

    def fingauss(self, R, pm=None):
        if pm is None:
            pm = self.pm

        kny = numpy.pi*pm.Nmesh[0]/pm.BoxSize[0]
        def tf(k):
            k2 = sum(((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny)))**2  for ki in k)
            wts = numpy.exp(-0.5*k2* R**2)
            return wts
        return tf            


    def create_ivar3d(self, mapp, noisefile, noisevar, smooth=None, mnmin=0):
        '''Different noise at the position of data for discrete data
        Use 4th column of the file - file format is mhigh, mlow, mean, std
        '''
        ivar = self.pm.create(mode='real')
        noise = numpy.loadtxt(noisefile)
        #noise[:, -1] *=2 #Add hoc factor of 2 to increase noise! 
        noise3d = numpy.ones_like(mapp[...])*noisevar**0.5

        if smooth is not None:
            tf = self.fingauss(smooth) 
            mappsm = mapp.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            mappsm = mapp

        for foo in range(noise.shape[0]):
            #file format is mhigh, mlow, mean, std
            mhigh = noise[foo][0]
            mlow = noise[foo][1]
            pos = numpy.where((mappsm[...] > mlow) & (mappsm[...] < mhigh))
            #Not smaller noise than empty points!
            #if noisevar**0.5 < noise[foo][3]:
            #    noise3d[pos] = noise[foo][3]
            if mhigh > mnmin:
                noise3d[pos] = noise[foo][3]
            #if self.pm.comm.rank == 0: print(mhigh, mlow, len(pos), noise[foo][2])

        ivar[...] = noise3d**2
        #save_map(self.mapp, path, 'mapp')

        self.ivar3d = ivar ** -1


    def create_off3d(self, mapp, noisefile, smooth=None):
        '''Mean offset, using 3rd column of the file- file format is mhigh, mlow, mean, std
        '''
        self.offset = self.pm.create(mode='real')
        #offset array should have data-model, so subtract from residual i.e. model-data
        noise = numpy.loadtxt(noisefile)
        mean3d = numpy.zeros_like(mapp[...])

        if smooth is not None:
            tf = self.fingauss(smooth) 
            mappsm = mapp.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            mappsm = mapp

        # for foo in range(len(mbinsm) -2):
        for foo in range(noise.shape[0]):
            mhigh = noise[foo][0]
            mlow = noise[foo][1]
            pos = numpy.where((mappsm[...] > mlow) & (mappsm[...] <= mhigh))
            mean3d[pos] = noise[foo][2]
            #if self.pm.comm.rank == 0: print(mhigh, mlow, len(pos), noise[foo][2])
        self.offset[...] = mean3d


    def downsample(self, pm):
        d = NoiseModel(pm, None, self.power, self.seed)
        d.mask2d = d.pm2d.downsample(self.mask2d)
        return d

#    def downsample3d(self, pm):
#        d = NoiseModel(pm, None, self.power, self.seed)
#        d.mask2d = d.pm2d.downsample(self.mask2d)
#        return d

    def add_noise(self, obs):
        pm = self.pm

        if self.seed is None:
            n = pm.create(mode='real')
            n[...] = 0
        else:
            n = pm.generate_whitenoise(mode='complex', seed=self.seed)
            n = n.apply(lambda k, v : (self.power / pm.BoxSize.prod()) ** 0.5 * v, out=Ellipsis).c2r(out=Ellipsis)
        return Observable(mapp=obs.mapp + n, s=obs.s, d=obs.d)

class Objective(base.Objective):
    def __init__(self, mock_model, noise_model, data, prior_ps, M0):
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        self.M0 = M0

    def get_code(self):
        pm = self.mock_model.pm

        code = base.Objective.get_code(self)

        data = self.data.mapp
        M0 = self.M0

        #likelihood is in log(M + M0)
        logdataM0 = self.pm.create(mode = 'real')
        logdataM0.value[...] = numpy.log(data + M0)
        code.add(x1='model', x2=Literal(M0), y='modelM0')
        code.log(x='modelM0', y='logmodelM0')
        code.add(x1='logmodelM0', x2=Literal(logdataM0*-1.), y='residual')

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    def __init__(self, mock_model, noise_model, data, prior_ps, sml, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps, M0=M0)
        self.sml = sml
        self.noised = noised
        self.smooth = smooth
        self.M0 = M0
        self.L1 = L1
        self.offset = offset
        self.smoothp = smoothprior

    def get_code(self):
        import numpy
        pm = self.mock_model.pm

        code = self.mock_model.get_code()
        data = self.data.mapp

        #sm
        if self.smooth is not None:
            code.r2c(real='model', complex='d_km')
            code.fingauss_smoothing(smoothed='modelsm', R=self.smooth, d_k='d_km')
            #
            def fingauss(pm, R):
                kny = numpy.pi*pm.Nmesh[0]/pm.BoxSize[0]
                def tf(k):
                    k2 = sum(((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny)))**2  for ki in k)
                    wts = numpy.exp(-0.5*k2* R**2)
                    return wts
                return tf            
            tf = fingauss(pm, self.smooth) 
            data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            code.assign(x='model', y='modelsm')

        #likelihood is M/M0
        M0 = self.M0
        if self.pm.comm.rank == 0:
            print('M0 is - %0.3e\n'%self.M0)
        logdataM0 = self.pm.create(mode = 'real')
        logdataM0.value[...] = data/M0
        code.multiply(x1='modelsm', x2=Literal(1/M0), y='logmodelM0')
        code.add(x1='logmodelM0', x2=Literal(logdataM0*-1.), y='residual')

        if self.offset:
            #offset array has data-model, so add to residual i.e. model-data
            if pm.comm.rank == 0:
                print('Offset in the noise \n')
            code.add(x1='residual', x2=Literal(self.noise_model.offset), y='residual')

        if self.noised == 2:
            if pm.comm.rank == 0:
                print('2D noise model\n\n')
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        elif self.noised == 3:
            if pm.comm.rank == 0:
                print('3D noise model\n\n')
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar3d ** 0.5), y='residual')

        #Fuck up right here....this was uncommented until Dec 4!
        #code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        
        #Smooth
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))

        code.r2c(real='residual', complex='C')
        code.transfer(complex='C', tf=smooth_window)
        code.c2r(real='residual', complex='C')
        #LNorm
        if self.L1:
            if pm.comm.rank == 0:
                print('L1 norm objective\n\n')
            code.L1norm(x='residual', y='chi2')
        else:
            if pm.comm.rank == 0:
                print('L2 norm objective\n\n')
            code.to_scalar(x='residual', y='chi2')

        #Prior
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        ## Smooth prior as well ##
        if self.smoothp:
            if pm.comm.rank == 0:
                print('Smooth Prior')
            code.r2c(real='pvar', complex='CC')
            code.transfer(complex='CC', tf=smooth_window)
            code.c2r(real='pvar', complex='CC')

        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code





def infoparse(ppath, ofolder=None):
    ifile = open(ppath + 'info.log')
    for line in ifile:
        if 'Features' in line:
            lf = line
        if 'architecture' in line:
            ln = line
    ifile.close()
    ftname = lf.split("'")[1::2]
    acts = []
    for s in ['relu', 'sigmoid', 'linear']:
        acts +=[(s, k.span()[0]) for k in re.finditer(s, ln)]    
    acts.sort(key = lambda tup: tup[1])
    acts = [i[0] for i in acts]
    try:
        true_width = [int(ln[k.span()[0]-1]) for k in re.finditer('x', ln)][0]
    except:
        true_width = 1
    
    if ofolder is not None:
        ifile = open(ppath + 'info.log')
        ofile = open(ofolder + 'inforead.log', 'w')
        for line in ifile:
            ofile.write(line)
        ofile.close()
        ifile.close()
    return ftname, acts, true_width
