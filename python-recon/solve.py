import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel,
                        LBFGS, ParticleMesh)
from cosmo4d.lab import dg
from abopt.algs.lbfgs import scalar as scalar_diag
import os
from time import time

#optimizer
def solve(Nmesh, x0, rtol, run, Nsm, prefix, mock_model, obj, data_p, truth_pm, optfolder, saveit=20, showit=5, title=None, maxiter=100, linesearchiter=100):
    
    pm = truth_pm.resize(Nmesh=(Nmesh, Nmesh, Nmesh))
    atol = pm.Nmesh.prod() * rtol
    #x0 = pm.upsample(x0, keep_mean=True)
    #data = data_n.downsample(pm)
    #Artificially Low Noise right now
    #data = data_p.downsample(pm)
    data = data_p
 
    
    #noise_model = truth_noise_model.downsample(pm)
  
    prior, chi2 = obj.get_code().compute(['prior', 'chi2'], init={'parameters': data.s})
    if pm.comm.rank == 0:
        print('\nPrior and chi2 at data.s \n',  prior, chi2) # for 2d chi2 is close to total pixels.
 
    #Evaluate and save 0 point
    fit_p = mock_model.make_observable(data.s)
    r = dg.evaluate(fit_p, data)
 
    try:
        os.makedirs(optfolder + '%s' % run)
    except:
        pass
    try:
        os.makedirs(optfolder + '%s/2pt' % run)
    except:
        pass
    
    if pm.comm.rank == 0:
        print('Currently output-ing in folder \n%s\n\n'%(optfolder + run))

    dg.save_report(r, optfolder + "%s/truth.png" % run, pm)
    dg.save_2ptreport(r, optfolder + "%s/2pt/truth.png" % run, pm)
 
    optimizer = LBFGS(m=10, diag_update=scalar_diag, maxiter=maxiter, linesearchiter=linesearchiter)
 
    prob = obj.get_problem(atol=atol, precond=UseComplexSpaceOptimizer)
 
    if saveit < showit: 
        showit = saveit
        if pm.comm.rank == 0: print('Setting showit = saveit = %d'%saveit)
    
    def monitor(state):
        if pm.comm.rank == 0:
            tstart = time()
            print(state)
            #print('Time for 1 step : ', time() - tstart)
        if state.nit % showit == 0:
            fit_p = mock_model.make_observable(state['x'])
            if state.nit % saveit == 0:
                fit_p.save(optfolder + '%s/%04d/fit_p' % (run, state['nit']))
            r = dg.evaluate(fit_p, data)
            #obj.save_report(r, optfolder + "%s/%s_N%02d-%04d.png"% (run, prefix, int(Nsm*10), state['nit']))
            dg.save_report(r, optfolder + "%s/%s_N%02d-%03d.png"% (run, prefix, int(Nsm*10), state['nit']), pm, title)
            dg.save_2ptreport(r, optfolder + "%s/2pt/%s_N%02d-%03d.png"% (run, prefix, int(Nsm*10), state['nit']), pm, title)
            if pm.comm.rank == 0:
                print('saved')
 
 
    state = optimizer.minimize(prob, x0=x0, monitor=monitor)
    fit_p = mock_model.make_observable(state['x'])
    fit_p.save(optfolder + '%s/best-fit' % run)
    r = dg.evaluate(fit_p, data)
    dg.save_report(r, optfolder + "%s/best-fit.png" % run, pm, title)
    dg.save_2ptreport(r, optfolder + "%s/2pt/%s_N%02d-best-fit.png" % (run, prefix, int(Nsm*10)), pm, title)
    return state.x



