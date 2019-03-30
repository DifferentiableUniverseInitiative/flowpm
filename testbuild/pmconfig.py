import numpy as np
import numpy
from nbodykit.lab import BigFileMesh, BigFileCatalog
from pmesh.pm import ParticleMesh
from tfpmfuncsdev import fftk
from nbodykit.cosmology import Cosmology, EHPower, Planck15

##
class Config(dict):
    def __init__(self, bs=40., nc=8, seed=100, B=1, dtype=np.float32):

        self['dtype'] = dtype
        self['boxsize'] = bs
        self['shift'] = 0.0
        self['nc'] = int(nc)
        self['ndim'] = 3
        self['seed'] = seed
        self['pm_nc_factor'] = B
        self['resampler'] = 'cic'
        self['cosmology'] = Planck15
        self['powerspectrum'] = EHPower(Planck15, 0)
        self['unitary'] = False
        self['stages'] = numpy.linspace(0.1, 1.0, 5, endpoint=True)
        self['aout'] = [1.0]
        self['kvec'] = fftk(shape=(nc, nc, nc), boxsize=bs, symmetric=False, dtype=dtype)
        self['grid'] = bs/nc*np.indices((nc, nc, nc)).reshape(3, -1).T.astype(dtype)

        local = {} # these names will be usable in the config file
        local['EHPower'] = EHPower
        local['Cosmology'] = Cosmology
        local['Planck15'] = Planck15
        local['linspace'] = numpy.linspace
#         local['autostages'] = autostages

        import nbodykit.lab as nlab
        local['nlab'] = nlab

        self.finalize()

    def finalize(self):
        self['aout'] = numpy.array(self['aout'])

        self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'], dtype='f4')
        mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
        missing_stages = self['aout'][mask]
        if len(missing_stages):
            raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))

        bs, nc, B = self['boxsize'], self['nc'], self['pm_nc_factor']
        ncf = int(nc*B)
        self['f_config'] = {'boxsize':bs, 
                            'nc':int(ncf), 
                            'kvec':fftk(shape=(ncf, ncf, ncf), boxsize=bs, symmetric=False, dtype=self['dtype'])}
