import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.interpolate import interp1d
from scipy.integrate import romberg

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


#######
#For galaxies


def csmfcen(m, mh=None, mstarc=1e12, sigc=0.212):
    '''Eq. 3 in Reddick et.al. 10.1088/0004-637X/771/1/30'''

    if mh is not None:
        mstarc = mstarcen(mh)
    fac = 1/np.sqrt(2*np.pi*sigc**2)
    expf = -(np.log10(m) - np.log10(mstarc))**2/(2*sigc**2)
    return fac * np.exp(expf)
    

    
def csmfsat(m, mh=None, mstars=1e12, alpha=-1, phis=1):
    '''Eq. 4 in Reddick et.al. 10.1088/0004-637X/771/1/30'''
    if mh is not None:
        mstars = mstarsat(mh)
        phis = phistar(mh)
    frac = m/mstars
    return phis * frac ** (alpha+1) * np.exp(-frac)


def mstarcen(mh):
    '''Eq. 5 and table 4, VAGC values from Reddick et.al'''
    m0 = 10**10.64
    m1 = 10**12.59
    g1 = 0.726
    g2 = 0.065
    logm = np.log10(m0) + g1*np.log10(mh/m1) + (g2-g1)*np.log10(1+mh/m1)
    return 10**logm


def phistar(mh):
    '''Eq. 6 and table 4, VAGC values from Reddick et.al'''
    mphi = 10**12.30
    a = 0.866
    return (mh/mphi)**a


def mstarsat(mh):
    '''Eq. 7 and table 4, VAGC values from Reddick et.al'''
    m0 = 10**10.401
    m1 = 10**12.71
    b = 0.753
    logm = np.log10(m0) + b*np.log10(mh/m1) - b*np.log10(1+mh/m1)
    return 10**logm



def createicdf(pdf=None, mv=None, mfv=None, Mthresh=10**10.5, M0=1e13):
    '''For the galaxies, based on the functions from Reddick et.al above
    Create a cdf for a halo mass and then sample from ssampleicdf
    - pdf (f) should be with the differential of log10(M)
    '''

    Nint = 500 #No.of points to interpolate
    if pdf is None:
        if mfv is None:
            print('need pdf or the values to interpolate over')
            return None
        else:
            imf = interpolate(mv, mfv, k = 5)
    else: imf = pdf
    ilmf = lambda x: imf(10**(x))
    lmm = np.linspace(np.log10(M0), np.log10(Mthresh), Nint)
      
    #Calculate the other mass-limit M2 of the catalog by integrating mf and comparing total number of halos
    nbin = abs(np.array([romberg(ilmf, lmm[i], lmm[i+1]) for i in range(lmm.size-1)]))
    ncum = np.cumsum(nbin)
    ncum /= ncum.max()
    lmm = 0.5*(lmm[:-1] + lmm[1:])
    if (ncum==0).sum() > 0: 
        zeros = np.where(ncum ==0)[0]
        ncum = ncum[zeros[-1]:]
        lmm = lmm[zeros[-1]:]
    if (ncum==1).sum() > 0: 
        ones = np.where(ncum ==1)[0]
        ncum = ncum[:ones[0]+1]
        lmm = lmm[:ones[0]+1]
    #try: icdf = interpolate(ncum, lmm)
    try: icdf = interp1d(ncum, lmm)
    except Exception as e: 
        print(e)
        print(ncum, lmm)
    return icdf


def sampleicdf(icdf, N0, seed=11, sort=True):
    np.random.seed(seed)
    ran = np.random.uniform(0, 1, N0)
    hmatch = 10**(icdf(ran))
    if sort:
        hmatch.sort()
        hmatch = hmatch[::-1]
    
    return hmatch

        


def assigngalaxymass(galcat, hmass, mbins=None, nmbins=10, verbose=False, seed=100, Mthresh=10**10.5, M0=1e13, 
                     sortcen=True, sortsat=True, cenpars=None, satpars=None, mstarc=1e12, sigc=0.212):
    '''
    Assign central and satellite masses based on matching the conditional stellar mass
    function from reddick et. al.
    For halos in given mass bins (mbins), generate csmf and then match it by inverse
    cdf sampling
    '''
    galtype = galcat['gal_type'].compute()
    haloid = galcat['halo_id'].compute()
    if verbose:
        haloidcen = set(haloid[galtype==0])
        haloidsat = set(haloid[galtype==1])
        satonly = haloidsat - haloidcen
        print('Number of halos with only satellites and no centrals = ', len(satonly))
    
    #hngal = np.zeros_like(hmass).astype('int')
    #unique, counts = np.unique(haloid, return_counts=True)
    #hngal[unique.astype('int')] = counts

    if mbins is None:
        mbins  = np.logspace(np.log10(hmass.min()), np.log10(hmass.max()), nmbins)[::-1]

    galmass = np.zeros_like(galtype).astype('float64')

    #Loop
    if sigc is False:
        print('No scatter')
        cens = galtype == 0
        sats = galtype == 1
        mhcens = hmass[haloid[cens]]
        mhsats = hmass[haloid[sats]]
        galmass[cens] = mstarcen(mhcens)
        galmass[sats] = mstarsat(mhsats)

        
    else:
        for i in range(len(mbins)-1):
            mh = mbins[i] + mbins[i+1]
            mh /= 2
            try: r1, r2 = np.where(hmass < mbins[i])[0][0], np.where(hmass <= mbins[i+1])[0][0]
            except: r1, r2 = np.where(hmass < mbins[i])[0][0], hmass.size
            if verbose: print("For halos in rank : ", r1, r2)

            mask = (haloid >= r1) & (haloid < r2)
            cens = mask & (galtype == 0)
            sats = mask & (galtype == 1)

            if verbose:
                ngal = hngal[r1:r2].sum()
                print('mask sum, #cen, #sat, #gal = ', mask.sum(), cens.sum(), sats.sum(), ngal)
                if ngal != cens.sum() + sats.sum():
                    print('wtf, ngal != cen+sat')


            mm = np.logspace(8, np.log10(mh))

            #Centrals
            #mfv = csmfcen(mm, mh = mh)
            #icdfc = sampler(M0=mh, mv=mm, mfv=mfv)
            icdfc = createicdf(pdf = lambda x: csmfcen(x, mh=mh, mstarc=mstarc, sigc=sigc), Mthresh=Mthresh, M0=mh)
            galmass[cens] = sampleicdf(icdfc, cens.sum(), seed=seed*i, sort=sortcen)

            #Satellites
            mfv = csmfsat(mm, mh = mh)
            icdfs = createicdf(mv=mm, mfv=mfv, Mthresh=Mthresh, M0=mh)
        #     icdfs = sampler(pdf = lambda x: icdf.csmfsf(x, mh=mh), M0=mh)
            galmass[sats] = sampleicdf(icdfs, sats.sum(), seed=seed*i, sort=sortsat)

    return galmass
