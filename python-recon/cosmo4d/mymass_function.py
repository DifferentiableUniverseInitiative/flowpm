import numpy
import numpy as np
import math

from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.integrate import simps
from scipy.integrate import romberg

import sys
sys.path.append("/global/homes/c/chmodi/Programs/Py_codes/modules/")
import mycosmology as cosmo_lib

def stellar_mass(hmass):
    '''Stellar to halo mass relationship taken from Moster et.al
    http://iopscience.iop.org/article/10.1088/0004-637X/710/2/903/pdf
    Eq. 2, values from Table 1

    '''
    lm1 = 11.884
    m1 = 10**lm1
    m0 = 0.02820
    b = 1.057
    g = 0.556
    smass = 2*hmass*m0* ((hmass/m1)**-b + (hmass/m1)**g)**-1
    return smass


#def fmexp(hmass, exp=0.45, c=10**5.27):
def fmexp(hmass, exp=None, cc=None):
#    if exp is None: exp=0.5
#    if cc is None: cc=4.6
    if exp is None: exp=1.0
    if cc is None: cc=0.0
    return 10**cc*hmass**exp
    

class Mass_Func():

    def __init__(self, power_file, M, L = None, H0 = 100.):
        
        self.M = M
        if L is None:
            L = 1- M
        self.L = L
        self.ktrue, self.ptrue = numpy.loadtxt(power_file, unpack = True)
        self.H0 = H0
        self.rhoc =  3 * H0**2 /(8 * math.pi * 43.007)
        self.rhom = self.rhoc*M
        self.cosmo = cosmo_lib.Cosmology(M= M, L = L, pfile = power_file)
        self.masses = 10**numpy.arange(9, 18, 0.01)
        self.sigma = numpy.zeros_like(self.masses)
        self.calc_sigma()
        self.fmap = {'Mice':self.Micef, 'ST':self.STf, 'Watson':self.Watsonf, 'Press':self.Pressf}

    def calc_sigma(self):
        M = self.masses
        for foo in range(len(M)):
            self.sigma[foo] = self.sigmasq(M[foo])**0.5


    def tophat(self, k, R):    
        kr = k*R
        wt = 3 * (numpy.sin(kr)/kr - numpy.cos(kr))/kr**2
        if wt is 0:
            wt = 1
        return wt

    def rLtom(self, R):
        """returns Mass in solar mass for smoothing scale in Mpc"""
        m = 4* math.pi*self.rhom * R**3 /3.
        return m*10**10

    def mtorL(self, m):
        """returns lagrangian radius (in Mpc) for Mass in solar mass"""
        rhobar = self.rhom * 10**10
        R3 = m /(4* math.pi*rhobar /3.)
        return R3**(1/3.)

    def sm_scale(self, M):    
        """returns smoothing scale in Mpc for Mass in solar mass"""
        return self.mtorL(M)
    #rhobar = self.rhom * 10**10
    #   R3 = 3* M /(4 * math.pi * rhobar)
    #  return R3 ** (1/3.)

    def sigmasq(self, M):
        """returns sigma**2 corresponding to mass in  solar mass"""
        R = self.sm_scale(M)
        k = self.ktrue
        p = self.ptrue
        w2 = self.tophat(k, R)**2
        return simps(p * w2 * k**2, k)/2/math.pi**2


    def dlninvsigdM(self, M, sigmaf = None, a = 1.):
        """ returns d(ln(1/sigma))/d(M) for M in solar masses. Can specify redshift with scale factor"""

        if sigmaf is None:
            sigmaf = self.sigmaf
        dM = 0.001 * M
        Mf = M + dM/2.
        Mb = M - dM/2.
        lnsigf = numpy.log(1/sigmaf(a)(Mf))
        lnsigb = numpy.log(1/sigmaf(a)(Mb))
        return (lnsigf - lnsigb)/dM

    def sigmaf(self, a = 1.):
        """ returns interpolating function for sigma. syntax to use - sigmaf(a)(M)"""
        d = self.cosmo.Dgrow(a = a)
        return interpolate(self.masses, d*self.sigma)

    def Micef(self, M, a=1., dndm = True):
        s = self.sigmaf(a)(M)
        z = self.cosmo.atoz(a)
        if a== 1 or z==0:
            A = 0.58
            a = 1.37
            b = 0.3
            c = 1.036
        elif a==0.66 or z==0.5:
            A = 0.55
            a = 1.29
            b = 0.29
            c = 1.026
        f = A *(s ** -a + b) * numpy.exp(- c / s**2)
        if dndm:
            return f * self.rhom * self.dlninvsigdM(M, self.sigmaf, a = a) *10**10
        else:
            return f

    def Tinkerf(self, M, a=1., dndm = True, delta =200.):
        s = self.sigmaf(a)(M)
        if delta == 200:
            A, a, b, c = 0.186, 1.47, 2.57, 1.19
        elif delta == 300:
            A, a, b, c = 0.200, 1.52, 2.25, 1.27
            
        f = A*((s/b)**-a + 1)*numpy.exp(-c/s**2.)
        if dndm:
            return f * self.rhom * self.dlninvsigdM(M, self.sigmaf, a = a) *10**10
        else:
            return f


    def Watsonf(self, M, a=1., dndm = True):
        s = self.sigmaf(a)(M)
        A = 0.282
        a = 2.163
        g = 1.210
        b = 1.406
        f = A *((b/s) ** a + 1) * numpy.exp(- g/ s**2)
        if dndm:
            return f * self.rhom * self.dlninvsigdM(M, self.sigmaf, a = a) *10**10
        else:
            return f

    def Pressf(self, M, a = 1., dndm = True):
        delc = 1.686
        nu = delc/self.sigmaf(a)(M)
        f = numpy.sqrt(2/math.pi) * nu * numpy.exp(- (nu**2) /2.) 
        if dndm:
            return f * self.rhom * self.dlninvsigdM(M, self.sigmaf, a = a) *10**10
        else:
            return f


    def STf(self, M, a = 1., dndm = True):
        delc = 1.686
        nu = delc/self.sigmaf(a)(M)
        a = 0.75
        p = 0.3
        f = 0.3222* numpy.sqrt(2*a/math.pi) * nu * numpy.exp(- a *(nu**2) /2.)*( 1 + 1/(a * nu**2)**p)
        if dndm:
            return f * self.rhom * self.dlninvsigdM(M, self.sigmaf, a = a) *10**10
        else:
            return f


    def DnDlnm(self, M, mfunc = 'Mice', a = 1):
        mf = self.fmap[mfunc]
        return self.rhom * mf(M, a = a, dndm = False) * self.dlninvsigdM(M, a = a) *10**10

    def match_abundance(self, halomass, bs, mfunc = 'Mice', a = 1, Mmin = 10.**11., Mmax = None):
        '''Returns new halomasses by matching abundance to given mass function'''
        if Mmax is None:
            Mmax = halomass[0]*1.01
        marray = numpy.exp(numpy.arange(numpy.log(Mmin), numpy.log(Mmax), 0.01))
        abund = []
        l = marray.shape[0]
        # f = lambda x:temp_mice(x, a)
        f = lambda x:self.DnDlnm(numpy.exp(x), mfunc = mfunc, a = a)
        for foo in range(0, marray.shape[0]):
            abund.append(romberg(f, numpy.log(marray)[l-foo-1], numpy.log(marray)[-1]))

        abund = numpy.array(abund)
        nexpect = abund*bs**3
        newmass = interpolate(nexpect, marray[::-1])
        halomassnew = newmass(numpy.linspace(1, len(halomass), num = len(halomass), endpoint=True))
        return halomassnew


    def icdf_sampling(self, bs, mfunc='Mice', match_high = True, hmass = None, M0 = None, N0 = None, seed=100, z=0):
        '''
        Given samples from analytic mass function (dN/dln(M)), find halo masss by matching abundance via
        inverse cdf sampling. 
        bs : boxsize
        mv, mfv : (Semi-optional) analytic hmf sampled at masses 'mv'
        mf : (Semi-optional) Analytic hmf, if mv and mfv are not given
        match_high : if True, Match the highest mass of the catalog to analytic mass.
        if False, match the lowest mass
        hmass : (Semi-Optional) Halo mass catalog, used to calculate highest/lowest mass 
        and number of halos
        M0, N0 : (Semi-optional) If mass catalog not given, M0 and N0 are required to 
        correspond to highest/lowest mass and number if halos
        
        Returns: Abundance matched halo mass catalog
        '''
        Nint = 500 #No.of points to interpolate
        a = self.cosmo.ztoa(z)
        if z!=0 and z!=0.5: 
            print('\nNOTE: Mass function changeed to Watson\n')
            mfunc = 'Watson'
        mf = self.fmap[mfunc]
        mv = np.logspace(10, 17, Nint)
        mfv = mf(mv, a=a)
        #Interpolate
        imf = interpolate(mv, mfv, k = 5)
        ilmf = lambda x: imf(np.exp(x))
    
        #Create array to integrate high or low from the matched mass based on match_high
        if N0 is None:
            N0 = hmass.size
        if match_high:
            if M0 is None:
                M0 = hmass.max()
            lmm = np.linspace(np.log(M0), np.log(mv.min()), Nint)
        else:
            if M0 is None:
                M0 = hmass.min()
            lmm = np.linspace(np.log(M0), np.log(mv.max()), Nint)

        #Calculate the other mass-limit M2 of the catalog by integrating mf and comparing total number of halos
        ncum = abs(np.array([romberg(ilmf, lmm[0], lmm[i]) for i in range(lmm.size)]))*bs**3
        M2 = np.exp(np.interp(N0, ncum, lmm))
        
        #Create pdf and cdf for N(M) from mf between M0 and M2
        lmm2 = np.linspace(np.log(M0), np.log(M2), Nint)
        nbin = abs(np.array([romberg(ilmf, lmm2[i], lmm2[i+1]) for i in range(lmm2.size-1)]))*bs**3
        nprob = nbin/nbin.sum()
        cdf = np.array([nprob[:i+1].sum() for i in range(nprob.size)])
        icdf = interpolate(cdf[:], 0.5*(lmm2[:-1] + lmm2[1:]))
        
        #Sample random points from uniform distribution and find corresponding mass
        np.random.seed(seed)
        ran = np.random.uniform(0, 1, N0)
        hmatch = np.exp(icdf(ran))
        hmatch.sort()
        return hmatch[::-1]

    def stellar_mass(self, hmass):
        return stellar_mass(hmass)

    def fmexp(self, hmass, exp=None, cc=None):
        return fmexp(hmass, exp, cc)




class Num_Mass_Func():

    def __init__(self, bs, nc, M, L = None, H0 = 100):

        self.M = M
        if L is None:
            L = 1- M
        self.L = L
        self.H0 = H0
        self.bs = float(bs)
        self.nc = nc
        self.rhoc =  3 * H0**2 /(8 * math.pi * 43.007)
        self.rhom = self.rhoc*M
        self.mp = self.rhom *(self.bs/self.nc)**3 * 10.**10
        self.vol = self.bs**3
        self.lMin = numpy.log10(1. * 10**11)
        self.lMax = numpy.log10(5. * 10**15)
        self.dlM = 0.05

#    def calc(self, halofile, lMin = 0, lMax = 0, dlM = 0):
##    def calc_file(self, halofile, lMin, lMax, dlM, warren = 0):
##        '''Calculate numerical mass function by binning in logspace between lMin, lMax with dlm.
##        Returns mf, counts, Mmean for every bin'''
##
##        import h5py
##        a = h5py.File(halofile, "r")
##        halo = a["FOFGroups"][:]
##        halomass = halo["Length"][1:]
##        if warren:
##            halomass = halomass *(1 - halomass**-0.6)
##        halomass = halomass*self.mp
##        
##        return self.calc_array(halomass, lMin, lMax, dlM)
##
    def calc_array(self, halomass, lMin, lMax, dlM):
        '''Calculate numerical mass function by binning in logspace between lMin, lMax with dlm.
        Returns mf, counts, Mmean for every bin'''

        nMbins = int((lMax - lMin)/dlM)
        
        lmass = numpy.zeros(nMbins)
        for foo in range(nMbins):
            lmass[foo] = lMin + foo*dlM

        counts = numpy.zeros((nMbins - 1))
        Mmean = numpy.zeros((nMbins - 1))
        
        ranks = numpy.zeros(nMbins)
        lsorthalomass = numpy.log10(halomass)[::-1]

        ### This is the magic counter to count number of halos in bins 
        for foo in range(nMbins):
            ranks[foo] = numpy.searchsorted(lsorthalomass, lmass[foo])
        for foo in range(nMbins - 1):
            counts[foo] = ranks[foo + 1] - ranks[foo]
        for foo in range(nMbins - 1):
            Mmean[foo] = (10**(lsorthalomass[int(ranks[foo]):int(ranks[foo+ 1])])).mean()

        mf = counts/self.vol/dlM/numpy.log(10)

        return mf, counts, Mmean
    
    
#nMbins = int((lMax - lMin)/dlM)
#        
#        lmass = numpy.zeros(nMbins)
#        for foo in range(nMbins):
#            lmass[foo] = lMin + foo*dlM
#
#        counts = numpy.zeros((nMbins - 1))
#        Mmean = numpy.zeros((nMbins - 1))
#        
#        ranks = numpy.zeros(nMbins)
#        lsorthalomass = numpy.log10(halomass)[::-1]
#
#        ### This is the magic counter to count number of halos in bins 
#        for foo in range(nMbins):
#            ranks[foo] = numpy.searchsorted(lsorthalomass, lmass[foo])
#        for foo in range(nMbins - 1):
#            counts[foo] = ranks[foo + 1] - ranks[foo]
#        for foo in range(nMbins - 1):
#            Mmean[foo] = (10**(lsorthalomass[int(ranks[foo]):int(ranks[foo+ 1])])).mean()
#
#        mf = counts/self.vol/dlM/numpy.log(10)
#
#        return mf, counts, Mmean
#    




def icdf_sampling(bs, mf = None, mv = None, mfv = None, match_high = True, hmass = None, M0 = None, N0 = None, seed=100):
    '''
    Given samples from analytic mass function (dN/dln(M)), find halo masss by matching abundance via
    inverse cdf sampling. 
    bs : boxsize
    mv, mfv : (Semi-optional) analytic hmf sampled at masses 'mv'
    mf : (Semi-optional) Analytic hmf, if mv and mfv are not given
    match_high : if True, Match the highest mass of the catalog to analytic mass.
        if False, match the lowest mass
    hmass : (Semi-Optional) Halo mass catalog, used to calculate highest/lowest mass 
        and number of halos
    M0, N0 : (Semi-optional) If mass catalog not given, M0 and N0 are required to 
        correspond to highest/lowest mass and number if halos
        
    Returns: Abundance matched halo mass catalog
    '''
    Nint = 500 #No.of points to interpolate
    #Create interpolating function for mass_func
    if mf is not None:
        mv = np.logspace(10, 17, Nint)
        mfv = mf(mv)
    elif mv is None:
            print("Need either a function or values sampled from the analytic mass function to match against")
            return None
    #Interpolate
    imf = interpolate(mv, mfv, k = 5)
    ilmf = lambda x: imf(np.exp(x))
    
    #Create array to integrate high or low from the matched mass based on match_high
    if N0 is None:
        N0 = hmass.size
    if match_high:
        if M0 is None:
            if hmass is None:
                print("Need either a halo mass catalog or a mass to be matched at")
                return 0
            else:
                M0 = hmass.max()
        lmm = np.linspace(np.log(M0), np.log(mv.min()), Nint)
    else:
        if M0 is None:
            M0 = hmass.min()
        lmm = np.linspace(np.log(M0), np.log(mv.max()), Nint)

        
    #Calculate the other mass-limit M2 of the catalog by integrating mf and comparing total number of halos
    ncum = abs(np.array([romberg(ilmf, lmm[0], lmm[i]) for i in range(lmm.size)]))*bs**3
    M2 = np.exp(np.interp(N0, ncum, lmm))

    #Create pdf and cdf for N(M) from mf between M0 and M2
    lmm2 = np.linspace(np.log(M0), np.log(M2), Nint)
    nbin = abs(np.array([romberg(ilmf, lmm2[i], lmm2[i+1]) for i in range(lmm2.size-1)]))*bs**3
    nprob = nbin/nbin.sum()
    cdf = np.array([nprob[:i+1].sum() for i in range(nprob.size)])
    icdf = interpolate(cdf[:], 0.5*(lmm2[:-1] + lmm2[1:]))
    
    #Sample random points from uniform distribution and find corresponding mass
    np.random.seed(seed)
    ran = np.random.uniform(0, 1, N0)
    hmatch = np.exp(icdf(ran))
    hmatch.sort()
    return hmatch[::-1]

