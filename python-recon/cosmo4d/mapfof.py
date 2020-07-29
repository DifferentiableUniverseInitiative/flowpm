### Module to generate the data for optimization i.e. FOF halo mass field
###


import numpy
from . import base
from .lab import NBodyModel, ParticleMesh
from .engine import Literal
from .iotools import save_map, load_map
from nbodykit.lab import FieldMesh, BigFileCatalog
from nbodykit.algorithms.fof import FOF
from nbodykit.lab import KDDensity, ArrayCatalog
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
import os

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




class FoFModel(base.MockModel):
    def __init__(self, dynamic_model, fofcat, numd, mf=None, sigma=None, abund=True, seed=100, ofolder=None, zz=0., mexp=None, stellar=False, cc=0.0, rsd=False, truemass=False, posonly=False, roundpos=None):
        self.dynamic_model = dynamic_model
        self.pm = dynamic_model.pm
        self.fofcat = fofcat
        self.numd = numd
        self.sigma = sigma
        self.seed = seed
        self.abund = abund
        self.mf = mf
        self.ofolder=ofolder
        self.zz = zz
        self.mexp = mexp
        self.cc = cc
        self.stellar = stellar
        self.rsd = rsd
        self.truemass = truemass         #upon scatter, whether to use true mass or not
        self.rsdfactor = self._rsdfac()
        self.posonly = posonly
        self.roundpos = roundpos


    def _rsdfac(self):
        aa = self.mf.cosmo.ztoa(self.zz)
        ff = self.mf.cosmo.Fomega1(self.mf.cosmo.ztoa(self.zz))
        rsdfac = 100/(aa**2 * self.mf.cosmo.Ha(z=self.zz)**1)
        #rsdfac *= 3 ######################################
        if self.pm.comm.rank ==0: print('\nrsd factor for velocity in data_mapfof is = %0.2f\n'%rsdfac)
        return rsdfac

    def get_code(self):

        pm, fofcat = self.pm, self.fofcat
        rank = fofcat.comm.rank
        code = self.dynamic_model.get_code()
        num = int(self.numd * pm.BoxSize.prod())
        savetup = ['CMPosition', 'CMVelocity', 'PeakPosition', 'PeakVelocity', 'Length', 'Mass']

        #Scatter; then match mass-function; then use matched mass
##        if self.sigma is not None:
##            #scatter here
##            rank, i0, i1 = fofcat.comm.rank, fofcat.Index.compute()[0], fofcat.Index.compute()[-1]
##
##            numpy.random.seed(seed = self.seed)
##            scatter = numpy.random.normal(scale=self.sigma, size=fofcat.csize)
##
##            logl = numpy.log10(fofcat['Mass'].compute()*1e10)
##            logl += scatter[i0:i1+1]
##            scmass = 10**logl
##
##            fofcat['scattered'] = scmass/1e10
##            fofcat = fofcat.sort('scattered', reverse=True)
##            savetup.append('scattered')
##
##        #match abundance
##        fofcat = fofcat.gslice(start = 1, stop = num+1)
##        fofcat  = fofcat.sort('Mass', reverse=True)
##
##        #match mass func
##        if self.abund:
##            i00, i11 = fofcat.Index.compute()[0], fofcat.Index.compute()[-1]            
##            M0 = fofcat.gslice(0, 1)['Mass'].compute()*1e10
##            M0 = numpy.concatenate(fofcat.comm.allgather(M0))[0]
##            hmassicdf = self.mf.icdf_sampling(pm.BoxSize[0], M0 = M0, N0 = fofcat.csize)
##            fofcat['AMass'] = hmassicdf[i00:i11+1]/1e10
##            mass = fofcat['AMass'].compute()*1e10
##            savetup.append('AMass')
##        else:
##            mass = fofcat['Mass'].compute()*1e10
##
##
        #match mass-function; then scatter; use scattered mass
        #match mass func


        fofcat  = fofcat.sort('Mass', reverse=True)
        if self.abund:
            i00, i11 = fofcat.Index.compute()[0], fofcat.Index.compute()[-1]            
            M0 = fofcat.gslice(0, 1)['Mass'].compute()*1e10
            M0 = numpy.concatenate(fofcat.comm.allgather(M0))[0]

            if pm.comm.rank == 0:
                print('Maximum mass for abundance matching = %0.3e'%M0)

            hmassicdf = self.mf.icdf_sampling(pm.BoxSize[0], M0 = M0, N0 = fofcat.csize, z=self.zz)
            fofcat['AMass'] = hmassicdf[i00:i11+1]/1e10
            usekey = 'AMass'
            savetup.append('AMass')
        else:
            usekey = 'Mass'

        if self.sigma is not None:
            #scatter here
            if pm.comm.rank == 0:
                print('Scattering catalog with scatter = %0.2f'%self.sigma)

            rank, i0, i1 = fofcat.comm.rank, fofcat.Index.compute()[0], fofcat.Index.compute()[-1]
            numpy.random.seed(seed = self.seed)
            scatter = numpy.random.normal(scale=self.sigma, size=fofcat.csize)

            logl = numpy.log10(fofcat[usekey].compute()*1e10)
            logl += scatter[i0:i1+1]
            scmass = 10**logl

            fofcat['scattered'] = scmass/1e10
            fofcat = fofcat.sort('scattered', reverse=True)
            savetup.append('scattered')
            if self.truemass: 
                if self.abund: 
                    usekey = 'AMass'
                else: usekey = 'Mass'
            else: usekey = 'scattered'
            
##
##        #scaling to stellar mass
##        if self.mexp is not None:
##            if not rank:
##                print('Scaling with mexp = %0.2f '%(self.mexp))
##            #mstar = 10**self.cc*(fofcat[usekey].compute()*1e10)**self.mexp
##            mstar = (fofcat[usekey].compute()*1e10)**self.mexp
##            fofcat['Mexp'] = mstar /1e10
##            usekey = 'Mexp'
##            savetup.append('Mexp')
##
##        if self.cc is not None and self.cc is not 0:
##            if not rank:
##                print('Scaling with  cc = %0.2f'%(self.cc))
##            if self.mexp is None: mstar = (fofcat[usekey].compute()*1e10)
##            mstar = 10**self.cc*mstar
##            fofcat['Mexp'] = mstar /1e10
##            usekey = 'Mexp'
##            savetup.append('Mexp')
##
##

        if self.stellar:
            mstar = self.mf.stellar_mass(fofcat[usekey].compute()*1e10)
            fofcat['Mstellar'] = mstar /1e10
            usekey = 'Mstellar'
            savetup.append('Mstellar')
            
        if self.rsd:
            position = fofcat['PeakPosition'].compute()
            velocity = fofcat['CMVelocity'].compute()
            vz = velocity*numpy.array([0, 0, 1]) * self.rsdfactor
            rsposition = position + vz
            fofcat['RSPeakPosition'] = rsposition
            savetup.append('RSPeakPosition')

            position = fofcat['CMPosition'].compute()
            #velocity = fofcat['CMVelocity'].compute()
            #vz = velocity*numpy.array([0, 0, 1])
            rsposition = position + vz
            fofcat['RSCMPosition'] = rsposition
            savetup.append('RSCMPosition')

        if self.ofolder is not None:
            fofcat.save(self.ofolder + 'FOFall', tuple(savetup))

        #match abundance
        #fofcat = fofcat.gslice(start = 1, stop = num+1)
        fofcat = fofcat.gslice(start = 0, stop = num)
        mass = fofcat[usekey].compute()*1e10
        if pm.comm.rank == 0:
            print('\n Key used to get mass is --- %s \n\n'%usekey)


        if self.ofolder is not None:
            if pm.comm.rank == 0: print('Saving the data in FOFd')
            fofcat.save(self.ofolder + 'FOFd', tuple(savetup))

        if self.rsd:
            if pm.comm.rank == 0: print('\n RSD! Key used to get position is --- RSPeakPosition \n\n')            
            position = fofcat['RSPeakPosition'].compute()
        else:
            position = fofcat['PeakPosition'].compute()
            
        #mass = fofcat['Mass'].compute()*1e10
        #print('In rank %d, Maximum mass = %0.3e'%(pm.comm.rank, mass.max()))
        if pm.comm.rank == 0:
            print('Number of halos used and those found in rank 0 = ', num, mass.size)

        layout = pm.decompose(position)
        if self.posonly:
            if pm.comm.rank==0:print('\nOnly position, no mass weighing\n')
            if self.roundpos is not None:
                if pm.comm.rank==0:print('\nRounding off at r=%0.2f\n'%self.roundpos)
                print(self.roundpos)
                #mesh = pm.paint(position)[...].flatten()
                #mesh[mesh > self.roundpos] = 1
                #grid = self.pm.generate_uniform_particle_grid(shift=0)
                #print(grid.shape, mesh.size, mass.size, position.shape, layout.indices.size)
                #code.paint(x=Literal(grid), mesh='model', layout=Literal(layout), mass=Literal(mesh))
                #code.paint(x=Literal(grid), mesh='model',  mass=Literal(mesh))
                code.paintdirect(x=Literal(position), mesh='model', layout=Literal(layout))
                code.logistic(x='model', y='model', t=self.roundpos, w=100)
            else:
                if pm.comm.rank==0:print('\nNo rounding off\n')
                code.paintdirect(x=Literal(position), mesh='model', layout=Literal(layout))
        else:
            if pm.comm.rank==0:print('\nMass weighing with key = %s\n'%usekey)
            code.paintdirect(x=Literal(position), mesh='model', layout=Literal(layout), mass=Literal(mass))

        return code

    def make_observable(self, initial):
        code = self.get_code()
        model, final = code.compute(['model', 'final'], init={'parameters':initial})
        return Observable(mapp=model, s=initial, d=final)





class GalModel(base.MockModel):
    def __init__(self, dynamic_model, galcat,  ofolder=None, zz=0., mf=None, rsd=False):
        self.dynamic_model = dynamic_model
        self.pm = dynamic_model.pm
        self.galcat = galcat
        self.mf = mf
        self.ofolder=ofolder
        self.zz = zz
        self.rsd = rsd
        self.rsdfactor = self._rsdfac()

    def _rsdfac(self):
        aa = self.mf.cosmo.ztoa(self.zz)
        ff = self.mf.cosmo.Fomega1(self.mf.cosmo.ztoa(self.zz))
        rsdfac = 100/(aa**2 * self.mf.cosmo.Ha(z=self.zz)**1)
        #rsdfac *= 3 ######################################
        if self.pm.comm.rank ==0: print('\nrsd factor for velocity in data_mapfof is = %0.2f\n'%rsdfac)
        return rsdfac

    def get_code(self):

        pm, galcat = self.pm, self.galcat
        rank = galcat.comm.rank
        code = self.dynamic_model.get_code()
        savetup = [col for col in galcat.columns]#['CMPosition', 'CMVelocity', 'PeakPosition', 'PeakVelocity', 'Length', 'Mass']

        galcat  = galcat.sort('Mass', reverse=True)
        usekey = 'Mass'      
        
#            
#        if self.rsd:
#            position = galcat['Position'].compute()
#            velocity = galcat['Velocity'].compute()
#            vz = velocity*numpy.array([0, 0, 1]) * self.rsdfactor
#            rsposition = position + vz
#            galcat['RSPeakPosition'] = rsposition
#            savetup.append('RSPeakPosition')
#
#            position = galcat['CMPosition'].compute()
#            #velocity = galcat['CMVelocity'].compute()
#            #vz = velocity*numpy.array([0, 0, 1])
#            rsposition = position + vz
#            galcat['RSCMPosition'] = rsposition
#            savetup.append('RSCMPosition')
#
        if self.ofolder is not None:
            galcat.save(self.ofolder + 'Galcat', tuple(savetup))

        #
        mass = galcat[usekey].compute()
        if pm.comm.rank == 0:
            print('\n Key used to get mass is --- %s \n\n'%usekey)

        if self.rsd:
            if pm.comm.rank == 0: print('\n RSD! Key used to get position is --- RSPeakPosition \n\n')            
            position = galcat['RSPeakPosition'].compute()
        else:
            position = galcat['Position'].compute()
            

        layout = pm.decompose(position)
        code.paintdirect(x=Literal(position), mesh='model', layout=Literal(layout), mass=Literal(mass))

        return code

    def make_observable(self, initial):
        code = self.get_code()
        model, final = code.compute(['model', 'final'], init={'parameters':initial})
        return Observable(mapp=model, s=initial, d=final)


#######################

def make_finer(bs, nc, seed, nsteps, cosmo, pk, ofolder, zz=0):
    #initiate

    pmf = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')

    try:
        if pmf.comm.rank == 0:
            print('Trying to read existing catalog from folder \n%s'%ofolder)
        fofcat = BigFileCatalog(ofolder + 'FOF', header='Header')
    
    except:
        print('File not found in rank  = ', pmf.comm.rank)

        try:
            os.makedirs(ofolder)
        except:
            pass
        if pmf.comm.rank == 0:
            print('Creating a new simulation with box, mesh, seed = %d,  %d, %d'%(bs, nc, seed))
            print('pk works at k = 0.01, p = %0.2f'%pk(0.01))


        s_truth = pmf.generate_whitenoise(seed, mode='complex')\
                          .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
                          .c2r()
##        s_truth = pmf.generate_whitenoise(seed, mode='complex')\
##                .apply(lambda k, v: v * (pk(sum(ki **2 for ki in k) **0.5) / v.BoxSize.prod()) ** 0.5)\
##                .c2r()
##
        if pmf.comm.rank == 0:
            print('truth generated')

        #dynamics
        aa = 1.0/(1+zz)
        stages = numpy.linspace(0.1, aa, nsteps, endpoint=True)
        dynamic_model = NBodyModel(cosmo, pmf, B=2, steps=stages)

        if pmf.comm.rank == 0:
            print('dynamic model created')

        print('Starting sim')

        X, V, final = dynamic_model.get_code().compute(['X', 'V', 'final'], init={'parameters':s_truth})

        if pmf.comm.rank == 0:
            print('X, Y, final computed')

        save_map(s_truth, ofolder + 'mesh', 's')
        save_map(final, ofolder + 'mesh', 'd')

        cat = ArrayCatalog({'Position': X, 'Velocity' : V}, BoxSize=pmf.BoxSize, Nmesh=pmf.Nmesh)
        kdd = KDDensity(cat).density
        cat['KDDensity'] = kdd
        cat.save(ofolder + 'dynamic/1', ('Position', 'Velocity', 'KDDensity'))
        if pmf.comm.rank == 0:
            print('High-res dynamic model created')

        #FOF
        fof = FOF(cat, linking_length=0.2, nmin=12)
        fofcat = fof.find_features(peakcolumn='KDDensity')
        fofcat['Mass'] = fofcat['Length'] * cosmo.Om0 * 27.7455 * pmf.BoxSize.prod() / pmf.Nmesh.prod()

        fofile = ofolder+'FOF'
        fofcat.save(fofile, ('CMPosition', 'CMVelocity', 'PeakPosition', 'PeakVelocity', 'Length', 'Mass'))
        if pmf.comm.rank == 0:
            print('Halos found in high-res simulation')
            print('Number of halos found in rank 0 = ', fofcat['Mass'].size)



    return fofcat


###    pmf = ParticleMesh(BoxSize=bs, Nmesh=(nc, nc, nc), dtype='f8')

###    filefound = None
###    if pmf.comm.rank == 0:
###        ffile = ofolder + '/FOF'
###        filefound = os.path.exists(ffile)
###        
###        print('Trying to read existing catalog from \n%s'%ffile)
###        print('Filefound in IF -', filefound, pmf.comm.rank)
###        fofcat = BigFileCatalog(ffile, header='Header')
###        if not filefound:
###            print('No!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
###            #import sys
###            #sys.exit('No!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
###        #filefound = pmf.comm.bcast(filefound, root=0)
###        
###
###    print('filefound before bcast  = ',filefound, pmf.comm.rank)
###    filefound = pmf.comm.bcast(filefound, root=0)
###    #filefound = pmf.comm.bcast(filefound, root=0)
###    print('filefound after bcast  = ',filefound, pmf.comm.rank)
###
###    if filefound is not None:
###        fofcat = BigFileCatalog(ofolder + '/FOF', header='Header')
###

