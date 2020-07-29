## Main model- use NN to generate halo mass field and optimize using 
## loss function of log(M_R + M_0) form.
## Noise and offset can be 3d 

import numpy
#from . import base
from cosmo4d import base
from cosmo4d.engine import Literal
from cosmo4d.iotools import save_map, load_map
from nbodykit.lab import FieldMesh
import re, json, warnings



class NoiseModel(base.NoiseModel):
    def __init__(self, pm, mask2d, power, seed):
        self.pm = pm
        #self.pm2d = self.pm.resize([self.pm.Nmesh[0], self.pm.Nmesh[1], 1])
        #if mask2d is None:
        #    mask2d = self.pm2d.create(mode='real')
        #    mask2d[...] = 1.0
        #self.mask2d = mask2d

        self.power = power
        self.seed = seed
        #self.var= power / (self.pm.BoxSize / self.pm.Nmesh).prod()
        #self.ivar2d = mask2d * self.var ** -1

        self.var = self.pm.create(mode='real', value=power / (self.pm.BoxSize / self.pm.Nmesh).prod())
        self.ivar2d = self.var ** -1

    #
    def fingauss(self, R, pm=None):
        if pm is None:
            pm = self.pm

        kny = numpy.pi*pm.Nmesh[0]/pm.BoxSize[0]
        def tf(k):
            k2 = sum(((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny)))**2  for ki in k)
            wts = numpy.exp(-0.5*k2* R**2)
            return wts
        return tf            


    def create_ivar3d(self, mapp, noisefile, noisevar, smooth=None):
        '''Different noise at the position of data for discrete data
        Use 4th column of the file - file format is mhigh, mlow, mean, std
        '''
        ivar = self.pm.create(mode='real')
        noise3d = numpy.ones_like(mapp[...])*noisevar**0.5

        if smooth is not None:
            tf = self.fingauss(smooth) 
            mappsm = mapp.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            mappsm = mapp

        if noisefile is not None:
            noise = numpy.loadtxt(noisefile)
            noise[:, -1] *=numpy.sqrt(2) #Add hoc factor of 2 to increase noise! 
            #As it happens, the factor of \sqrt(2) is cz of def of gaussian - \sqrt(2)*\sigma^2 in exp
            #Not sure about how to make it 2 instead?

    ##        for foo in range(noise.shape[0]):
    ##            #file format is mhigh, mlow, mean, std
    ##            mhigh = noise[foo][0]
    ##            mlow = noise[foo][1]
    ##            pos = numpy.where((mappsm[...] > mlow) & (mappsm[...] < mhigh))
    ##            #Not smaller noise than empty points!
    ##            if noisevar**0.5 < noise[foo][3]:
    ##                noise3d[pos] = noise[foo][3]
    ##            #if self.pm.comm.rank == 0: print(mhigh, mlow, len(pos), noise[foo][2])
    ##
            for foo in range(noise.shape[0]):
                #file format is mhigh, mlow, mean, std
                mhigh = noise[foo][0]
                mlow = noise[foo][1]
                pos = numpy.where((mappsm[...] > mlow) & (mappsm[...] < mhigh))
                noise3d[pos] = noise[foo][3]
        else:
            if self.pm.comm.rank == 0: print('\nWARNING: Asked to create 3D noise, but no noisefile is given\n')

        ivar[...] = noise3d**2
        #save_map(self.mapp, path, 'mapp')

        self.ivar3d = ivar ** -1



    def suppress_noise(self, mapp, mlim, ninf=1e3, smooth=None,  noisefile=None, autofac=2):
        '''Set noise below mlim to ninf where iinf is variance of noise
        '''

        if smooth is not None:
            tf = self.fingauss(smooth) 
            mappsm = mapp.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            mappsm = mapp

        if ninf == 'auto':
            noise = numpy.loadtxt(noisefile)
            noise[:, -1] *=numpy.sqrt(2) #Add hoc factor of 2 to increase noise! 
            noises = []
            for i in range(noise[:, 0].size):
                if noise[i+1, 0] >  mlim: 
                    noises.append(noise[:, -1])
                else: break
            maxnoise = numpy.array(noises).max()
            ninf = (maxnoise)**2
            ninf *= autofac
            if self.pm.comm.rank == 0: print('Varaince to suppress set to ninf = %0.3f'%ninf)

        if mlim is not None:
            #print('Number of points below mlim = %.2e is = '%mlim, (mappsm[...] < mlim).sum())
            pos = numpy.where((mappsm[...] < mlim))
            self.ivar2d[pos] = ninf**-1#-2
            try : 
                self.ivar3d[pos] = ninf**-1#-2
            except AttributeError: 
                if self.pm.comm.rank == 0: print('No 3D noise to suppress')


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

#    def add_noise(self, obs):
#        pm = self.pm
#
#        if self.seed is None:
#            n = pm.create(mode='real')
#            n[...] = 0
#        else:
#            n = pm.generate_whitenoise(mode='complex', seed=self.seed)
#            n = n.apply(lambda k, v : (self.power / pm.BoxSize.prod()) ** 0.5 * v, out=Ellipsis).c2r(out=Ellipsis)
#        return Observable(mapp=obs.mapp + n, s=obs.s, d=obs.d)
