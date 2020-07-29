
import numpy
#from . import base
from cosmo4d import base
from cosmo4d.engine import Literal
from cosmo4d.iotools import save_map, load_map
from nbodykit.lab import FieldMesh
import re, json, warnings



class Objective(base.Objective):
    def __init__(self, mock_model, noise_model, data, prior_ps, M0=0):
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        self.M0 = M0

    def get_code(self):
        pass

##    def evaluate(self, model, data):
##      pass




###########################



def fingauss(pm, R):
    kny = numpy.pi*pm.Nmesh[0]/pm.BoxSize[0]
    def tf(k):
        k2 = sum(((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny)))**2  for ki in k)
        wts = numpy.exp(-0.5*k2* R**2)
        return wts
    return tf            



class SmoothedObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml, smlmax):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.smlmax = smlmax

    def get_code(self):
        import numpy
        pm = self.mock_model.pm

        code = self.mock_model.get_code()

        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        if self.noise_model is not None: 
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        high_pass_window = lambda k: 1 - numpy.exp(- (self.sml + self.smlmax) ** 2 * sum(ki ** 2 for ki in k))
        highpass = False
        if self.smlmax is not None:
            if self.sml < self.smlmax:
                highpass = True
                if pm.comm.rank == 0: print('\nHigh pass filtering\n')
        code.transfer(complex='C', tf=smooth_window)
        if highpass : code.transfer(complex='C', tf=high_pass_window)
        code.c2r(real='residual', complex='C')
        
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='pvar', x2=Literal(pm.Nmesh.prod()**-1.), y='pvar')
        if highpass:
            code.r2c(real='pvar', complex='C2')
            code.transfer(complex='C2', tf=high_pass_window)
            code.c2r(real='pvar', complex='C2')

        code.to_scalar(x='residual', y='chi2')
        code.to_scalar(x='pvar', y='prior')
        code.add(x1='prior', x2='chi2', y='objective')

        return code



    
class SmoothedLogObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml, smlmax = None, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False):
        #Objective.__init__(self, mock_model, noise_model, data, prior_ps, M0=M0)
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model



        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        #
        self.sml = sml
        self.smlmax = smlmax
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
            tf = fingauss(pm, self.smooth) 
            data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            code.assign(x='model', y='modelsm')

        #likelihood is log(M+M0)
        M0 = self.M0
        if self.pm.comm.rank == 0:
            print('M0 is - %0.3e\n'%self.M0)
        logdataM0 = self.pm.create(mode = 'real')
        logdataM0.value[...] = numpy.log(data + M0)
        code.add(x1='modelsm', x2=Literal(M0), y='modelM0')
        code.log(x='modelM0', y='logmodelM0')
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
        high_pass_window = lambda k: 1 - numpy.exp(- (self.sml + self.smlmax) ** 2 * sum(ki ** 2 for ki in k))
        highpass = False
        if self.smlmax is not None:
            if self.sml < self.smlmax:
                highpass = True
                if pm.comm.rank == 0 : print("\n high pass filter")

        code.r2c(real='residual', complex='C')
        code.transfer(complex='C', tf=smooth_window)
        if highpass :
            code.transfer(complex='C', tf=high_pass_window)
        code.c2r(real='residual', complex='C')
        #LNorm
        if self.L1:
            if pm.comm.rank == 0: print('L1 norm objective\n\n')
            code.L1norm(x='residual', y='chi2')
        else:
            if pm.comm.rank == 0: print('L2 norm objective\n\n')
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

        if highpass :
            code.r2c(real='pvar', complex='CC')
            code.transfer(complex='CC', tf=high_pass_window)
            code.c2r(real='pvar', complex='CC')

        code.multiply(x1='pvar', x2=Literal(pm.Nmesh.prod()**-1.), y='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.add(x1='prior', x2='chi2', y='objective')
        return code






class SmoothedFourierObjective(Objective):
    """ The smoothed objecte smoothes the residual before computing chi2.
        It breaks the noise model at small scale, but the advantage is that
        the gradient in small scale is stronglly suppressed and we effectively
        only fit the large scale. Since we know usually the large scale converges
        very slowly this helps to stablize the solution.
    """
    def __init__(self, mock_model, noise_model, data, prior_ps, error_ps, sml, ivarmesh=None):
        Objective.__init__(self, mock_model, noise_model, data, prior_ps)
        self.sml = sml
        self.error_ps = error_ps
        self.ivarmesh = ivarmesh
        if self.ivarmesh is not None: self.ivarmeshc = self.ivarmesh.r2c()

    def get_code(self):
        import numpy
        pm = self.mock_model.pm

        code = self.mock_model.get_code()

        data = self.data.mapp

        code.add(x1='model', x2=Literal(data * -1), y='residual')
        code.r2c(real='residual', complex='C')
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
        code.transfer(complex='C', tf=smooth_window)
        if self.ivarmesh is None: code.create_whitenoise(dlinear_k='C', powerspectrum=self.error_ps, whitenoise='perror')
        else: 
            if pm.comm.rank == 0: print('Using ivarmesh')
            code.multiply(x1='C', x2=Literal(self.ivarmeshc**0.5), y='perrorc')
            code.c2r(complex='perrorc', real='perror')

        code.to_scalar(x='perror', y='chi2')
        #code.c2r(real='residual', complex='C')
        #code.to_scalar(x='residual', y='chi2')
        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
        code.to_scalar(x='pvar', y='prior')
        # the whitenoise is not properly normalized as d_k / P**0.5
        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
        code.add(x1='prior', x2='chi2', y='objective')
        return code





    

###########################################################################



class SmoothedLogOvdObjective(Objective):
    """ likelihood is log( (M+M0)/Mmean )

    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False):
        #Objective.__init__(self, mock_model, noise_model, data, prior_ps, M0=M0)
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        #
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
            tf = fingauss(pm, self.smooth) 
            data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            code.assign(x='model', y='modelsm')



        #likelihood is log( (M+M0)/Mmean )
        M0 = self.M0
        datamean = data.cmean()
        logdataM0 = numpy.log((data + M0)/datamean)

        inpoints = 1/pm.Nmesh.prod()
        code.total(x='modelsm', y='modeltot')
        code.multiply(x1='modeltot', x2=Literal(inpoints), y='modelmean')
        code.pow(x='modelmean', y='invmodelmean', power=-1)
        code.add(x1='modelsm', x2=Literal(M0), y='modelM0')
        code.multiply(x1='modelM0', x2='invmodelmean', y='modelM0')
        code.log(x='modelM0', y='logmodelM0')

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




###########################################################################



class SmoothedLogTotObjective(Objective):
    """ likelihood is log( (M+M0) ) + M_tot

    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False, wttotal=1000):
        #Objective.__init__(self, mock_model, noise_model, data, prior_ps, M0=M0)
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        #
        self.sml = sml
        self.noised = noised
        self.smooth = smooth
        self.M0 = M0
        self.L1 = L1
        self.offset = offset
        self.smoothp = smoothprior
        self.wttotal = wttotal

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
            tf = fingauss(pm, self.smooth) 
            data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else:
            code.assign(x='model', y='modelsm')



        #likelihood is log( (M+M0)/Mmean )
        M0 = self.M0
        datasum = data.csum()
        logdataM0 = numpy.log((data + M0))

        inpoints = 1/pm.Nmesh.prod()
        code.total(x='modelsm', y='modeltot')
        #code.multiply(x1='modeltot', x2=Literal(inpoints), y='modelmean')
        #code.pow(x='modelmean', y='invmodelmean', power=-1)
        code.add(x1='modelsm', x2=Literal(M0), y='modelM0')
        #code.multiply(x1='modelM0', x2='invmodelmean', y='modelM0')
        code.log(x='modelM0', y='logmodelM0')

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

        #Add the residual differece of the sum
        ##logdatasum = numpy.log(datasum)
        ##code.log(x='modeltot', y='logmodeltot')
        ##code.add(x1='logmodeltot', x2=Literal(-logdatasum), y='totaloffset')
        ##code.multiply(x1='totaloffset', x2='totaloffset', y='totaloffsetsq')
        ##code.multiply(x1='totaloffsetsq', x2=Literal(self.wttotal), y='chitotal')
        ##code.add(x1='objective', x2='chitotal', y='objective')
        ##
        code.add(x1='modelsm', x2=Literal(-data), y='totaloffset')
        code.multiply(x1='totaloffset', x2='totaloffset', y='totaloffsetsq')
        code.total(x='totaloffsetsq', y='chitotal')
        code.log(x='chitotal', y='logchitotal')
        code.multiply(x1='logchitotal', x2=Literal(self.wttotal), y='logchitotal')
        #code.total(x='totaloffsetsq', y='chitotal')
        #code.log(x='chitotal', y='logchitotal')
        code.add(x1='objective', x2='logchitotal', y='objective')
        
        return code



###########################################################################










class SmoothedOvdObjective(Objective):
    """ likelihood is M/(M0+Mmean)

    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False):
        #Objective.__init__(self, mock_model, noise_model, data, prior_ps, M0=M0)
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        #
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
            tf = fingauss(pm, self.smooth) 
            data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else: code.assign(x='model', y='modelsm')

        #likelihood is M/Mmean
        M0 = self.M0
        datamean = data.cmean()
        data[...] /= (M0 + datamean)

        inpoints = 1/pm.Nmesh.prod()
        code.total(x='modelsm', y='modeltot')
        code.multiply(x1='modeltot', x2=Literal(inpoints), y='modelmean')
        code.add(x1='modelmean', x2=Literal(M0), y='modelmean')
        code.pow(x='modelmean', y='invmodelmean', power=-1)
        code.multiply(x1='modelsm', x2='invmodelmean', y='modelsm')

        code.add(x1='modelsm', x2=Literal(data*-1.), y='residual')


        if self.offset:
            #offset array has data-model, so add to residual i.e. model-data
            if pm.comm.rank == 0: print('Offset in the noise \n')
            code.add(x1='residual', x2=Literal(self.noise_model.offset), y='residual')

        if self.noised == 2:
            if pm.comm.rank == 0: print('2D noise model\n\n')
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        elif self.noised == 3:
            if pm.comm.rank == 0: print('3D noise model\n\n')
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar3d ** 0.5), y='residual')

        
        #Smooth
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))

        code.r2c(real='residual', complex='C')
        code.transfer(complex='C', tf=smooth_window)
        code.c2r(real='residual', complex='C')

        #LNorm
        if self.L1:
            if pm.comm.rank == 0: print('L1 norm objective\n\n')
            code.L1norm(x='residual', y='chi2')
        else:
            if pm.comm.rank == 0: print('L2 norm objective\n\n')
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




###########################################################################




class SmoothedOvdM0Objective(Objective):
    """ likelihood is (M+M0)/(M0+Mmean)

    """
    def __init__(self, mock_model, noise_model, data, prior_ps, sml, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False):
        #Objective.__init__(self, mock_model, noise_model, data, prior_ps, M0=M0)
        self.prior_ps = prior_ps
        self.mock_model = mock_model
        self.noise_model = noise_model
        self.data = data
        self.pm = mock_model.pm
        self.engine = mock_model.engine
        #
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
            tf = fingauss(pm, self.smooth) 
            data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
        else: code.assign(x='model', y='modelsm')

        #likelihood is (M + M0)/Mmean
        M0 = self.M0
        data[...] += M0
        datamean = data.cmean()
        data[...] /= (datamean)

        inpoints = 1/pm.Nmesh.prod()
        code.add(x1='modelsm', x2=Literal(M0), y='modelsm')
        code.total(x='modelsm', y='modeltot')
        code.multiply(x1='modeltot', x2=Literal(inpoints), y='modelmean')
        #code.add(x1='modelmean', x2=Literal(M0), y='modelmean')
        code.pow(x='modelmean', y='invmodelmean', power=-1)
        code.multiply(x1='modelsm', x2='invmodelmean', y='modelsm')

        code.add(x1='modelsm', x2=Literal(data*-1.), y='residual')


        if self.offset:
            #offset array has data-model, so add to residual i.e. model-data
            if pm.comm.rank == 0: print('Offset in the noise \n')
            code.add(x1='residual', x2=Literal(self.noise_model.offset), y='residual')

        if self.noised == 2:
            if pm.comm.rank == 0: print('2D noise model\n\n')
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
        elif self.noised == 3:
            if pm.comm.rank == 0: print('3D noise model\n\n')
            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar3d ** 0.5), y='residual')

        
        #Smooth
        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))

        code.r2c(real='residual', complex='C')
        code.transfer(complex='C', tf=smooth_window)
        code.c2r(real='residual', complex='C')

        #LNorm
        if self.L1:
            if pm.comm.rank == 0: print('L1 norm objective\n\n')
            code.L1norm(x='residual', y='chi2')
        else:
            if pm.comm.rank == 0: print('L2 norm objective\n\n')
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




#@class SmoothedNoiseLogObjective(Objective):
#@    """ The smoothed objecte smoothes the residual before computing chi2.
#@        It breaks the noise model at small scale, but the advantage is that
#@        the gradient in small scale is stronglly suppressed and we effectively
#@        only fit the large scale. Since we know usually the large scale converges
#@        very slowly this helps to stablize the solution.
#@    """
#@    def __init__(self, mock_model, noise_model, data, prior_ps, sml, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False):
#@        #Objective.__init__(self, mock_model, noise_model, data, prior_ps, M0=M0)
#@        self.prior_ps = prior_ps
#@        self.mock_model = mock_model
#@        self.noise_model = noise_model
#@        self.data = data
#@        self.pm = mock_model.pm
#@        self.engine = mock_model.engine
#@        #
#@        self.sml = sml
#@        self.noised = noised
#@        self.smooth = smooth
#@        self.M0 = M0
#@        self.L1 = L1
#@        self.offset = offset
#@        self.smoothp = smoothprior
#@
#@    def get_code(self):
#@        import numpy
#@        pm = self.mock_model.pm
#@
#@        code = self.mock_model.get_code()
#@        data = self.data.mapp
#@
#@        #sm
#@        if self.smooth is not None:
#@            code.r2c(real='model', complex='d_km')
#@            code.fingauss_smoothing(smoothed='modelsm', R=self.smooth, d_k='d_km')
#@            #
#@            tf = fingauss(pm, self.smooth) 
#@            data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
#@        else:
#@            code.assign(x='model', y='modelsm')
#@
#@        #likelihood is log(M+M0)
#@        M0 = self.M0
#@        if self.pm.comm.rank == 0:
#@            print('M0 is - %0.3e\n'%self.M0)
#@        logdataM0 = self.pm.create(mode = 'real')
#@        logdataM0.value[...] = numpy.log(data + M0)
#@        code.add(x1='modelsm', x2=Literal(M0), y='modelM0')
#@        code.log(x='modelM0', y='logmodelM0')
#@        code.add(x1='logmodelM0', x2=Literal(logdataM0*-1.), y='residual')
#@
#@        if self.offset:
#@            #offset array has data-model, so add to residual i.e. model-data
#@            if pm.comm.rank == 0:
#@                print('Offset in the noise \n')
#@            code.add(x1='residual', x2=Literal(self.noise_model.offset), y='residual')
#@
#@        if self.noised == 2:
#@            if pm.comm.rank == 0:
#@                print('2D noise model\n\n')
#@            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
#@        elif self.noised == 3:
#@            if pm.comm.rank == 0:
#@                print('3D noise model\n\n')
#@            code.multiply(x1='residual', x2=Literal(self.noise_model.ivar3d ** 0.5), y='residual')
#@
#@        #Fuck up right here....this was uncommented until Dec 4!
#@        #code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
#@        
#@        #Smooth
#@        smooth_window = lambda k: numpy.exp(- self.sml ** 2 * sum(ki ** 2 for ki in k))
#@
#@        code.r2c(real='residual', complex='C')
#@        code.transfer(complex='C', tf=smooth_window)
#@        code.c2r(real='residual', complex='C')
#@        #LNorm
#@        if self.L1:
#@            if pm.comm.rank == 0:
#@                print('L1 norm objective\n\n')
#@            code.L1norm(x='residual', y='chi2')
#@        else:
#@            if pm.comm.rank == 0:
#@                print('L2 norm objective\n\n')
#@            code.to_scalar(x='residual', y='chi2')
#@
#@        #Prior
#@        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
#@        ## Smooth prior as well ##
#@        if self.smoothp:
#@            if pm.comm.rank == 0:
#@                print('Smooth Prior')
#@            code.r2c(real='pvar', complex='CC')
#@            code.transfer(complex='CC', tf=smooth_window)
#@            code.c2r(real='pvar', complex='CC')
#@
#@        code.to_scalar(x='pvar', y='prior')
#@        # the whitenoise is not properly normalized as d_k / P**0.5
#@        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
#@        code.add(x1='prior', x2='chi2', y='objective')
#@        return code







##
##class Objective(base.Objective):
##    def __init__(self, mock_model, noise_model, data, prior_ps, M0):
##        self.prior_ps = prior_ps
##        self.mock_model = mock_model
##        self.noise_model = noise_model
##        self.data = data
##        self.pm = mock_model.pm
##        self.engine = mock_model.engine
##        self.M0 = M0
##
##    def get_code(self):
##        pm = self.mock_model.pm
##
##        code = base.Objective.get_code(self)
##
##        data = self.data.mapp
##        M0 = self.M0
##
##        #likelihood is in log(M + M0)
##        logdataM0 = self.pm.create(mode = 'real')
##        logdataM0.value[...] = numpy.log(data + M0)
##        code.add(x1='model', x2=Literal(M0), y='modelM0')
##        code.log(x='modelM0', y='logmodelM0')
##        code.add(x1='logmodelM0', x2=Literal(logdataM0*-1.), y='residual')
##
##        code.multiply(x1='residual', x2=Literal(self.noise_model.ivar2d ** 0.5), y='residual')
##        code.to_scalar(x='residual', y='chi2')
##        code.create_whitenoise(dlinear_k='dlinear_k', powerspectrum=self.prior_ps, whitenoise='pvar')
##        code.to_scalar(x='pvar', y='prior')
##        # the whitenoise is not properly normalized as d_k / P**0.5
##        code.multiply(x1='prior', x2=Literal(pm.Nmesh.prod()**-1.), y='prior')
##        code.add(x1='prior', x2='chi2', y='objective')
##        return code
##
##    def evaluate(self, model, data):
##        from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower
##
##        xm = FFTPower(first=FieldMesh(model.mapp/model.mapp.cmean()), second=FieldMesh(data.mapp/data.mapp.cmean()), mode='1d')
##        xd = FFTPower(first=FieldMesh(model.d), second=FieldMesh(data.d), mode='1d')
##        xs = FFTPower(first=FieldMesh(model.s), second=FieldMesh(data.s), mode='1d')
##
##        pm1 = FFTPower(first=FieldMesh(model.mapp/model.mapp.cmean()), mode='1d')
##        pd1 = FFTPower(first=FieldMesh(model.d), mode='1d')
##        ps1 = FFTPower(first=FieldMesh(model.s), mode='1d')
##
##        pm2 = FFTPower(first=FieldMesh(data.mapp/data.mapp.cmean()), mode='1d')
##        pd2 = FFTPower(first=FieldMesh(data.d), mode='1d')
##        ps2 = FFTPower(first=FieldMesh(data.s), mode='1d')
##
##        data_preview = dict(s=[], d=[], mapp=[])
##        model_preview = dict(s=[], d=[], mapp=[])
##
##        for axes in [[1, 2], [0, 2], [0, 1]]:
##            data_preview['d'].append(data.d.preview(axes=axes))
##            data_preview['s'].append(data.s.preview(axes=axes))
##            data_preview['mapp'].append(data.mapp.preview(axes=axes))
##            model_preview['d'].append(model.d.preview(axes=axes))
##            model_preview['s'].append(model.s.preview(axes=axes))
##            model_preview['mapp'].append(model.mapp.preview(axes=axes))
##
##        #data_preview['mapp'] = data.mapp.preview(axes=(0, 1))
##        #model_preview['mapp'] = model.mapp.preview(axes=(0, 1))
##
##        return xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power, data_preview, model_preview
##
##    def save_report(self, report, filename):
##        xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2, data_preview, model_preview = report
##
##        km = xm['k']
##        ks = xs['k']
##        kd = xd['k']
##
##        with warnings.catch_warnings():
##            warnings.simplefilter("ignore")
##            xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
##            xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
##            xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5
##
##            tm = (pm1['power'] / pm2['power']) ** 0.5
##            ts = (ps1['power'] / ps2['power']) ** 0.5
##            td = (pd1['power'] / pd2['power']) ** 0.5
##
##        from cosmo4d.iotools import create_figure
##        fig, gs = create_figure((12, 9), (4, 6))
##        for i in range(3):
##            ax = fig.add_subplot(gs[0, i])
##            ax.imshow(data_preview['s'][i])
##            ax.set_title("s data")
##
##        for i in range(3):
##            ax = fig.add_subplot(gs[0, i + 3])
##            ax.imshow(data_preview['d'][i])
##            ax.set_title("d data")
##
##        for i in range(3):
##            ax = fig.add_subplot(gs[1, i])
##            ax.imshow(model_preview['s'][i])
##            ax.set_title("s model")
##
##        for i in range(3):
##            ax = fig.add_subplot(gs[1, i + 3])
##            ax.imshow(model_preview['d'][i])
##            ax.set_title("d model")
##
##        for i in range(3):
##            ax = fig.add_subplot(gs[2, i + 3])
##            ax.imshow(data_preview['mapp'][i])
##            ax.set_title("map data")
##
##        for i in range(3):
##            ax = fig.add_subplot(gs[3, i + 3])
##            ax.imshow(model_preview['mapp'][i])
##            ax.set_title("map model")
##
##        ax = fig.add_subplot(gs[2, 0])
##        ax.plot(ks, xs, label='intial')
##        ax.plot(kd, xd, label='final')
##        ax.plot(km, xm, label='map')
##        ax.legend()
##        ax.set_xscale('log')
##        ax.set_title('Cross coeff')
##
##        ax = fig.add_subplot(gs[2, 1])
##        ax.plot(ks, ts, label='intial')
##        ax.plot(kd, td, label='final')
##        ax.plot(km, tm, label='map')
##        ax.legend()
##        ax.set_xscale('log')
##        ax.set_title('Transfer func')
##
##        ax = fig.add_subplot(gs[2, 2])
##        ax.set_xlim(0, 1)
##        ax.set_ylim(0, 1)
##        ax.text(0.05, 0.9, 's=linear')
##        ax.text(0.05, 0.7, 'd=non-linear')
##        ax.text(0.05, 0.5, 'map=halos (sm)')
##        ax.text(0.05, 0.3, 'model=FastPM+NN')
##        ax.text(0.05, 0.1, 'data=FastPM+NN')
##
##        ax = fig.add_subplot(gs[3, 0])
##        ax.plot(ks, ps1['power'], label='model')
##        ax.plot(ks, ps2['power'], label='data')
##        ax.set_xscale('log')
##        ax.set_yscale('log')
##        ax.set_title("initial")
##        ax.legend()
##
##        ax = fig.add_subplot(gs[3, 1])
##        ax.plot(kd, pd1['power'], label='model')
##        ax.plot(kd, pd2['power'], label='data')
##        ax.set_xscale('log')
##        ax.set_yscale('log')
##        ax.set_title("final")
##        ax.legend()
##
##        ax = fig.add_subplot(gs[3, 2])
##        ax.plot(km, pm1['power'], label='model')
##        ax.plot(km, pm2['power'], label='data')
##        ax.set_xscale('log')
##        ax.set_yscale('log')
##        ax.set_title("map")
##        ax.legend()
##
##
##        fig.tight_layout()
##        if self.pm.comm.rank == 0:
##            fig.savefig(filename)
##
