import numpy as np
import numpy
from  astropy.cosmology import Planck15 
from background import *
##
class Config(dict):
    def __init__(self):

        self['boxsize'] = 40
        self['shift'] = 0.0
        self['nc'] = 8
        self['ndim'] = 3
        self['seed'] = 100
        self['pm_nc_factor'] = 1
        self['resampler'] = 'cic'
        self['cosmology'] = Planck15
        #self['powerspectrum'] = EHPower(Planck15, 0)
        self['unitary'] = False
        self['stages'] = numpy.linspace(0.1, 1.0, 5, endpoint=True)
        self['aout'] = [1.0]
        self['perturbation'] = MatterDominated(cosmo=self['cosmology'], a=self['stages'])
        
#        local = {} # these names will be usable in the config file
#        local['EHPower'] = EHPower
#        local['Cosmology'] = Cosmology
#        local['Planck15'] = Planck15
#        local['linspace'] = numpy.linspace
##         local['autostages'] = autostages


        self.finalize()

    def finalize(self):
        self['aout'] = numpy.array(self['aout'])

        #self.pm = ParticleMesh(BoxSize=self['boxsize'], Nmesh= [self['nc']] * self['ndim'], resampler=self['resampler'], dtype='f4')
        mask = numpy.array([ a not in self['stages'] for a in self['aout']], dtype='?')
        missing_stages = self['aout'][mask]
        if len(missing_stages):
            raise ValueError('Some stages are requested for output but missing: %s' % str(missing_stages))
