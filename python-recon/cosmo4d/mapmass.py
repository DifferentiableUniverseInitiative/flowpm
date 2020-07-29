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
        if not self.pm.comm.rank: print('Position dictionary is ', pdict)
        self.R1p = pdict['R1']
        try: self.R2p = pdict['R2']
        except: self.R2p = pdict['sfac']*self.R1p

        self.pftname = pdict['pftname']
        try:
            acts = pdict['activations']
            self.pacts = ['sigmoid' if 'sigmoid' in s else s for s in acts]
        except:
            self.pacts = ['relu', 'relu', 'sigmoid'] #Pytorch did not write activations
        if self.pwidth is None:
            self.pwidth = pdict['width']

        self.pwts, self.pbias = [], []

        for s in [0, 1, 2, 3, 4]:
            try:
                self.pwts.append(numpy.load(ppath + 'w%d.npy'%s))
                self.pbias.append(numpy.load(ppath + 'b%d.npy'%s))
            except:
                pass
        if 'torchnet' in ppath:
            if not self.pm.comm.rank: print('\nTorchnet in path, take transpose\n')
            for i in range(len(self.pwts)): 
                self.pwts[i] = self.pwts[i].T

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
        if not self.pm.comm.rank: print('Mass dictionary is ', pdict)

        self.mftname = pdict['mftname']
        try:
            self.macts = pdict['activations'] #Pytorch did not write activations
        except: 
            self.macts = ['elu', 'elu', 'linear']

        self.mwidth = 0
            
        self.mwts, self.mbias = [], []

        for s in [0, 1, 2, 3, 4]:
            try:
                self.mwts.append(numpy.load(mpath + 'w%d.npy'%s))
                self.mbias.append(numpy.load(mpath + 'b%d.npy'%s))
            except:
                pass
        if 'torchnet' in mpath:
            if not self.pm.comm.rank: print('\nTorchnet in path, take transpose\n')
            for i in range(len(self.mwts)): 
                self.mwts[i] = self.mwts[i].T

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
        ####Extra factor
        #code.relu(x='mpredict', y='mpredict')
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

