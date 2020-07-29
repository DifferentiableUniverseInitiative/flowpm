"""
This class takes in a cosmology from astropy and H0 value \
to define extra functions for that cosmology. 
|  If H0 is none, use it from the astropy-cosmology. Specify if you want to use 100. 
"""

import numpy
from scipy.integrate import romberg, quad
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate

from astropy.cosmology import FlatLambdaCDM

#To import other modules from this folder. 
#Figure out a better way to do it. Probably have to make a module or sth
import sys, os
pwd  = os.getcwd()
#modpath = "/global/homes/c/chmodi/Programs/Py_codes/modules/"
sys.path.append(pwd)
from halofit import halofit 


    
class Cosmology():
    #To use scale factor, explicitly specify a = ?. Implicitly, redshift is assumed.
    #def __init__(self, M = 0.3175, L = 0.6825, H0 = 100., B = None, klin = None, plin = None, pfile = modpath + 'ics_matterpow_0.dat'):
    def __init__(self, M = 0.3, L = None, H0 = 100., B = None, sig8 = 0.8, klin = None, plin = None, pfile = None):
        
        self.M = M
        if L is None:
            L = 1 - M
        self.L = L
        self.H0 = H0
        self.B = B
        self.sig8 = sig8
        self.cin = 1000./(299792458)
        
        self.pfile = pfile
        if pfile is not None:
            self.klin, self.plin = numpy.loadtxt(pfile, unpack = True)
        else:
            self.klin = klin
            self.plin = plin
        
        self.cosmo = FlatLambdaCDM(H0 = H0, Om0 = M, Ob0 = B, name = "cosmo")
        self.Dplus = self._eval_Dplus()
    
    def _za(self, z, a):
        if a is None:
            if z is None:
                print("Atleast one of scale factor or redshift is needed")
                return None
            else: 
                a = self.ztoa(z)
                return z, a
        else:
            z = self.atoz(a)
            return z, a
    
    def atoz(self, a):
        '''Convert scale factor to redshift'''
        return 1./a - 1.

    def ztoa(self, z):
        '''Convert redshift to scale factor'''
        return 1./(z+1)

    def Fomega1(self, a):
        """return \omega_m**(3./5..)"""
        M = self.M
        L = self.L
        H0 = self.H0

        omega_a = M / (M + (1 - M - L) * a + L * a ** 3)
        return omega_a ** (3./ 5.)

    def Fomega2(self, a):
        """return \omega_m**(4./7.)"""
        M = self.M
        L = self.L
        H0 = self.H0

        omega_a = M / (M + (1 - M - L) * a + L * a ** 3)
        return 2* omega_a ** (4./ 7)

    def _eval_Dplus(self):
        """return un-normalized D(a)
        ...evaluated only once because lazy"""
        M = self.M
        L = self.L
        H0 = self.H0
        logamin = -20.
        Np = 1000
        logx = numpy.linspace(logamin, 0, Np)
        x = numpy.exp(logx)

        def kernel(loga):
            a = numpy.exp(loga)
            return (a * self.Ea(a = a)) ** -3 * a # da = a * d loga

        y = self.Ea(a = x) * numpy.array([ romberg(kernel, logx.min(), loga, vec_func=True) for loga in logx])

        return interpolate(x,y)

    def Dgrow(self, z = None, a = None):
        """return D(a)/D(1.)"""
        z, a = self._za(z, a)
        return(self.Dplus(a)/self.Dplus(1.))


    def Ha(self, z = None, a = None):
        """return H(a)"""
        z, a = self._za(z, a)
        return(self.Ea(a = a)*self.H0)

    def Ea(self, z = None, a = None):
        """return H(a)/H0"""
        z, a = self._za(z, a)

        M, L = self.M, self.L
        return (a ** -1.5 * (M + (1 - M - L) * a + L * a ** 3) ** 0.5)


    def OmegaMa(self, z = None, a = None):
        """return Omega_m(a)"""
        M, L = self.M, self.L
        z, a = self._za(z, a)

        return M/a**3./(M/a**3. + L)

    def xia(self, z = None, a = None):
        """return comoving distance to scale factor a in Mpc/h"""
        fac = 100.*self.cin
        f = lambda x: (self.Ea(a = x)*x**2)**-1
        
        z, a = self._za(z, a)

        if type(a) == numpy.ndarray:
            y = numpy.zeros_like(a)
            for foo in range(len(a)):
                y[foo] = quad(f, a[foo], 1)[0]
            return y/fac
        else:
            return quad(f, a, 1)[0]/fac

    def k_xil(self, l, z = None, a = None):
        '''k-value corresponding to l at distance xi'''
        z, a = self._za(z, a)
        return l/self.xia(a = a)


    def phi_a(self):
        pass


    def pkalin(self, z = None, a = None):
        '''calculate linear power spectrum at scale factor 'a' '''
        z, a = self._za(z, a)        
        return self.klin, self.plin*self.Dgrow(a = a)**2.

    def pkanlin(self, z = None, a = None):
        '''calculate nonlinear power spectrum at scale factor 'a' from halofit'''
        z, a = self._za(z, a)        

        pdiml = self.plin*self.klin**3/(2.*numpy.pi**2)

        pgive = pdiml*self.Dgrow(a = a)**2.
        pndim = halofit(k = self.klin, delta_k=pgive, sigma_8 = self.sig8, z = self.atoz(a), cosmo=self.cosmo, takahashi=True)
        pnlin = pndim * (2*numpy.pi**2)/self.klin**3
        return self.klin, pnlin
        

        
    def ppota(self, z = None, a = None,  nlin = False):
        '''Calculate the power spectrum of the potential at scale factor a'''
        z, a = self._za(z, a)
        
        if nlin:
            pmat =  self.pkanlin(a = a)[1]
        else:
            pmat =  self.pkalin(a = a)[1]
        return self.klin, (9 * self.M**2 * self.H0**4 * pmat)/(4 * a**2 * self.klin**4) * self.cin**4
        #return [other form] (9 * csmo.OmegaMa(a)**2 * a**4 * csmo.Ha(a)**4 * pmat)/(4 * klin**4)
  



#class Lazy(object):
#    def __init__(self, calculate_function):
#        self._calculate = calculate_function
#
#    def __get__(self, obj, _=None):
#        if obj is None:
#            return self
#        value = self._calculate(obj)
#        setattr(obj, self._calculate.func_name, value)
#        return value


##    @Lazy
#    def _eval_Dplus(self):
#        """return un-normalized D(a)
#        ...evaluated only once because lazy"""
#        M = self.M
#        L = self.L
#        H0 = self.H0
#        logamin = -20.
#        Np = 1000
#        logx = numpy.linspace(logamin, 0, Np)
#        x = numpy.exp(logx)
#
#        def kernel(loga):
#            a = numpy.exp(loga)
#            return (a * self.Ea(a)) ** -3 * a # da = a * d loga
#
#        y = self.Ea(x) * numpy.array([ romberg(kernel, logx.min(), loga, vec_func=True) for loga in logx])
#
#        return interpolate(x,y)
