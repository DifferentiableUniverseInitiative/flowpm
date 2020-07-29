import numpy
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from cosmo4d.lab import (UseComplexSpaceOptimizer,
                        NBodyModel,
                        LBFGS, ParticleMesh)
from cosmo4d.lab import dg
from abopt.algs.lbfgs import scalar as scalar_diag
#from abopt.algs.gradient import GradientDescent
import os
from time import time
from abopt.abopt2 import LineSearchGradientDescent, GradientDescent, LBFGS, Preconditioner, minimize
from abopt.linesearch import minpack, backtrace, exact

#optimizer
def solve(Nmesh, x0, rtol, run, Nsm, prefix, mock_model, obj, data_p, truth_pm, optfolder, saveit=20, showit=5, title=None, maxiter=100, linesearchiter=100, rate=1.,
          opt='GD'):
    
  
    
    pm = truth_pm.resize(Nmesh=(Nmesh, Nmesh, Nmesh))
    atol = pm.Nmesh.prod() * rtol
    #data = data_p
 
    prior, chi2 = obj.get_code().compute(['prior', 'chi2'], init={'parameters': data_p.s})
    if pm.comm.rank == 0:
        print('\nPrior and chi2 at data.s \n',  prior, chi2) # for 2d chi2 is close to total pixels.
 
    #Evaluate and save 0 point
    fit_p = mock_model.make_observable(data_p.s)
    r = dg.evaluate(fit_p, data_p)
 
    try:        os.makedirs(optfolder + '%s' % run)
    except:        pass
    try:        os.makedirs(optfolder + '%s/2pt' % run)
    except:        pass
    
    if pm.comm.rank == 0:
        print('Currently output-ing in folder \n%s\n\n'%(optfolder + run))

    dg.save_report(r, optfolder + "%s/truth.png" % run, pm)
    dg.save_2ptreport(r, optfolder + "%s/2pt/truth.png" % run, pm)
 
    prior, chi2 = obj.get_code().compute(['prior', 'chi2'], init={'parameters': x0})
    if pm.comm.rank == 0:
        print('\nPrior and chi2 at x0 \n',  prior, chi2) # for 2d chi2 is close to total pixels.
 
    if opt == 'LBFGS': optimizer = LBFGS(m=10, diag_update=scalar_diag, maxiter=maxiter, linesearchiter=linesearchiter)
    if opt == 'LGD': optimizer = LineSearchGradientDescent(maxiter=maxiter, linesearchiter=linesearchiter, linesearch=backtrace)
    if opt == 'GD': optimizer = GradientDescent(maxiter=maxiter, rate=rate)
 
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
            r = dg.evaluate(fit_p, data_p)
            #obj.save_report(r, optfolder + "%s/%s_N%02d-%04d.png"% (run, prefix, int(Nsm*10), state['nit']))
            dg.save_report(r, optfolder + "%s/%s_N%02d-%03d.png"% (run, prefix, int(Nsm*10), state['nit']), pm)
            dg.save_2ptreport(r, optfolder + "%s/2pt/%s_N%02d-%03d.png"% (run, prefix, int(Nsm*10), state['nit']), pm)
            if pm.comm.rank == 0:
                print('saved')
 
 
    if pm.comm.rank == 0: print('Starting minimization ')
    state = optimizer.minimize(prob, x0=x0, monitor=monitor)
    fit_p = mock_model.make_observable(state['x'])
    fit_p.save(optfolder + '%s/best-fit' % run)
    r = dg.evaluate(fit_p, data_p)
    dg.save_report(r, optfolder + "%s/best-fit.png" % run, pm)
    dg.save_2ptreport(r, optfolder + "%s/2pt/%s_N%02d-best-fit.png" % (run, prefix, int(Nsm*10)), pm)
    return state.x



