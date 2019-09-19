import numpy as np
import numpy, os
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from  astropy.cosmology import Planck15
from background import MatterDominated, RadiationDominated
from tfpmfuncs import fftk

package_path = os.path.dirname(os.path.abspath(__file__))

class Config(dict):
    def __init__(self, bs=100., nc=32, seed=100, B=1, stages=None, cosmo=None, pkfile=None, pkinitfile=None, dtype=np.float32):

        self['dtype'] = dtype
        self['boxsize'] = dtype(bs)
        self['shift'] = 0.0
        self['nc'] = int(nc)
        self['kny'] = np.pi*nc/bs
        self['ndim'] = 3
        self['seed'] = seed
        self['pm_nc_factor'] = B
        self['resampler'] = 'cic'
        #
        self['cosmology'] = Planck15
        if stages is None: stages = numpy.linspace(0.1, 1.0, 5, endpoint=True)
        self['stages'] = stages
        self['aout'] = [1.0]
        self['perturbation'] = MatterDominated(cosmo=self['cosmology'], a=self['stages'])
        #self['perturbation'] = RadiationDominated(cosmo=self['cosmology'], a=self['stages'])
        #
        self['kvec'] = fftk(shape=(nc, nc, nc), boxsize=bs, symmetric=False, dtype=dtype)
        self['grid'] = bs/nc*np.indices((nc, nc, nc)).reshape(3, -1).T.astype(dtype)
        #
        #self['powerspectrum'] = EHPower(Planck15, 0)
        #self['unitary'] = False
        #
        if pkfile is None: pkfile = os.path.join(package_path , 'data/Planck15_a1p00.txt')
        self['pkfile'] = pkfile
        if pkinitfile is None: pkinitfile = os.path.join(package_path, '/Planck15_a0p01.txt')
        self['pkfile_ainit'] = pkinitfile

        self.finalize()


    def finalize(self):
        self['aout'] = numpy.array(self['aout'])
        self['klin'] = np.loadtxt(self['pkfile']).T[0]
        self['plin'] = np.loadtxt(self['pkfile']).T[1]
        self['ipklin'] = iuspline(self['klin'], self['plin'])
        #
        bs, nc, B = self['boxsize'], self['nc'], self['pm_nc_factor']
        ncf = int(nc*B)
        self['f_config'] = {'boxsize':bs,
                            'nc':int(ncf),
                            'kvec':fftk(shape=(ncf, ncf, ncf), boxsize=bs, symmetric=False, dtype=self['dtype'])}
        #
        #self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'], dtype='f4')
        mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
        missing_stages = self['aout'][mask]
        if len(missing_stages):
            raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))
