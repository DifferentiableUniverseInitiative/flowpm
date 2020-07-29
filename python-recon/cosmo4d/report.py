
import numpy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from cosmo4d.lab import mapfinal as mapp

def evaluate(model, data, M0=0, stoch=False, kmin=None, dk=None):
    '''return position order:
    xm.power, xs.power, xd.power, 
    pm1.power, pm2.power, 
    ps1.power, ps2.power, 
    pd1.power, pd2.power, 
    data_preview, model_preview
    '''
    from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower

    if kmin is None: kmin = 0 
    if dk is None: dk = 2*numpy.pi/model.s.BoxSize[0]

    model.mapp += M0
    data.mapp += M0


    #print('Means are : ', model.mapp.cmean(), data.mapp.cmean())
    if abs(model.mapp.cmean()) > 1e-3: modmappmean = model.mapp.cmean()
    else: modmappmean = 1.
    if abs(data.mapp.cmean()) > 1e-3: datmappmean = data.mapp.cmean()
    else: datmappmean = 1.
    modmappmean, datmappmean = 1., 1.
    #if abs(model.mapp.cmean()) > 1e-3: modmappmean = model.mapp.cmean()
    #else: modmappmean = 1.
    #if abs(data.mapp.cmean()) > 1e-3: datmappmean = data.mapp.cmean()
    #else: datmappmean = 1.
        
    xm = FFTPower(first=FieldMesh(model.mapp/modmappmean), second=FieldMesh(data.mapp/datmappmean), mode='1d', kmin=kmin, dk=dk)
    pm1 = FFTPower(first=FieldMesh(model.mapp/modmappmean), mode='1d', kmin=kmin, dk=dk)
    pm2 = FFTPower(first=FieldMesh(data.mapp/datmappmean), mode='1d', kmin=kmin, dk=dk)

    xs = FFTPower(first=FieldMesh(model.s), second=FieldMesh(data.s), mode='1d', kmin=kmin, dk=dk)
    ps1 = FFTPower(first=FieldMesh(model.s), mode='1d', kmin=kmin, dk=dk)
    ps2 = FFTPower(first=FieldMesh(data.s), mode='1d', kmin=kmin, dk=dk)

    xd = FFTPower(first=FieldMesh(model.d), second=FieldMesh(data.d), mode='1d', kmin=kmin, dk=dk)
    pd1 = FFTPower(first=FieldMesh(model.d), mode='1d', kmin=kmin, dk=dk)
    pd2 = FFTPower(first=FieldMesh(data.d), mode='1d', kmin=kmin, dk=dk)


    if stoch:
        psd = FFTPower(first=FieldMesh(data.s), second=FieldMesh(model.s), mode='1d', kmin=kmin, dk=dk)
        pdd = FFTPower(first=FieldMesh(data.d), second=FieldMesh(model.d), mode='1d', kmin=kmin, dk=dk)
        pmd = FFTPower(first=FieldMesh(data.mapp/data.mapp.cmean()), second=FieldMesh(model.mapp/model.mapp.cmean()), mode='1d', kmin=kmin, dk=dk)

    data_preview = dict(s=[], d=[], mapp=[])
    model_preview = dict(s=[], d=[], mapp=[])

    for axes in [[1, 2], [0, 2], [0, 1]]:
        data_preview['d'].append(data.d.preview(axes=axes))
        data_preview['s'].append(data.s.preview(axes=axes))
        data_preview['mapp'].append(data.mapp.preview(axes=axes))
        model_preview['d'].append(model.d.preview(axes=axes))
        model_preview['s'].append(model.s.preview(axes=axes))
        model_preview['mapp'].append(model.mapp.preview(axes=axes))

    #data_preview['mapp'] = data.mapp.preview(axes=(0, 1))
    #model_preview['mapp'] = model.mapp.preview(axes=(0, 1))

    if stoch:
        return xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power, \
        psd.power, pdd.power, pmd.power, data_preview, model_preview
        
    return xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power, data_preview, model_preview



def evaluate1(model, data, field, M0=0, stoch=False, kmin=None, dk=None):
    '''return position order:
    xm.power, xs.power, xd.power, 
    pm1.power, pm2.power, 
    ps1.power, ps2.power, 
    pd1.power, pd2.power, 
    data_preview, model_preview
    '''
    from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower

    if kmin is None: kmin = 0 
    if dk is None: dk = 2*numpy.pi/model.s.BoxSize[0]

    model.mapp += M0
    data.mapp += M0


    #print('Means are : ', model.mapp.cmean(), data.mapp.cmean())
    if abs(model.mapp.cmean()) > 1e-3: modmappmean = model.mapp.cmean()
    else: modmappmean = 1.
    if abs(data.mapp.cmean()) > 1e-3: datmappmean = data.mapp.cmean()
    else: datmappmean = 1.
    modmappmean, datmappmean = 1., 1.
    #if abs(model.mapp.cmean()) > 1e-3: modmappmean = model.mapp.cmean()
    #else: modmappmean = 1.
    #if abs(data.mapp.cmean()) > 1e-3: datmappmean = data.mapp.cmean()
    #else: datmappmean = 1.
        
    data_preview, model_preview = [], []
    if field == 'mapp':
        x = FFTPower(first=FieldMesh(model.mapp/modmappmean), second=FieldMesh(data.mapp/datmappmean), mode='1d', kmin=kmin, dk=dk)
        p1 = FFTPower(first=FieldMesh(model.mapp/modmappmean), mode='1d', kmin=kmin, dk=dk)
        p2 = FFTPower(first=FieldMesh(data.mapp/datmappmean), mode='1d', kmin=kmin, dk=dk)
        for axes in [[1, 2], [0, 2], [0, 1]]:
            data_preview.append(data.mapp.preview(axes=axes))
            model_preview.append(model.mapp.preview(axes=axes))

    elif field == 's':
        x = FFTPower(first=FieldMesh(model.s), second=FieldMesh(data.s), mode='1d', kmin=kmin, dk=dk)
        p1 = FFTPower(first=FieldMesh(model.s), mode='1d', kmin=kmin, dk=dk)
        p2 = FFTPower(first=FieldMesh(data.s), mode='1d', kmin=kmin, dk=dk)
        for axes in [[1, 2], [0, 2], [0, 1]]:
            data_preview.append(data.s.preview(axes=axes))
            model_preview.append(model.s.preview(axes=axes))

    elif field == 'd':
        x = FFTPower(first=FieldMesh(model.d), second=FieldMesh(data.d), mode='1d', kmin=kmin, dk=dk)
        p1 = FFTPower(first=FieldMesh(model.d), mode='1d', kmin=kmin, dk=dk)
        p2 = FFTPower(first=FieldMesh(data.d), mode='1d', kmin=kmin, dk=dk)
        for axes in [[1, 2], [0, 2], [0, 1]]:
            data_preview.append(data.d.preview(axes=axes))
            model_preview.append(model.d.preview(axes=axes))
    else:
        print('Field not recongnized')
        return 0
        
    return x.power, p1.power, p2.power, data_preview, model_preview

#
#def evaluate1(model, data, norm=True, kmin=None, dk=None):
#    '''return position order:
#    px,p1,p2
#    '''
#    from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower
#
#    if kmin is None: kmin = 0 
#    if dk is None: dk = 2*numpy.pi/model.BoxSize[0]
#
#    if norm:
#        mod = model/model.cmean()
#        dat = data/data.cmean()
#    else:
#        mod, dat = model, data
#
#    px = FFTPower(first=FieldMesh(mod), second=FieldMesh(dat), mode='1d', kmin=kmin, dk=dk)
#
#    p1 = FFTPower(first=FieldMesh(mod), mode='1d', kmin=kmin, dk=dk)
#
#    p2 = FFTPower(first=FieldMesh(dat), mode='1d', kmin=kmin, dk=dk)
#
#
#    return px.power, p1.power, p2.power
#



def create_2ptreport(report, filename=None, figin=None, axin=None):
    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2, data_preview, model_preview = report

    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
    xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
    xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

    tm = (pm1['power'] / pm2['power']) **0.5
    ts = (ps1['power'] / ps2['power']) **0.5
    td = (pd1['power'] / pd2['power']) **0.5

    from cosmo4d.iotools import create_figure
    if figin is None:
        fig, axar = plt.subplots(2, 3, figsize = (15, 8))
    else:
        fig, axar = figin, axin
    

    ax = axar[0, 0]
    #ax = fig.add_subplot(gs[0, 0])
    ax.plot(ks, xs, label='intial', ls = "-", lw=1.5)
    ax.plot(kd, xd, label='final', ls = "--", lw=1.5)
    ax.plot(km, xm, label='map', ls = ":", lw=2)
    ax.legend()
    ax.set_xscale('log')
    ax.set_title('Cross coeff')

    ax = axar[0, 1]
    #ax = fig.add_subplot(gs[0, 1])
    ax.plot(ks, ts, label='intial', ls = "-", lw=1.5)
    ax.plot(kd, td, label='final', ls = "--", lw=1.5)
    ax.plot(km, tm, label='map', ls = ":", lw=2)
    ax.legend()
    ax.set_xscale('log')
    ax.set_title('Transfer func')

    ax = axar[0, 2]
    #ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.05, 0.9, 's=linear')
    ax.text(0.05, 0.7, 'd=non-linear')
    ax.text(0.05, 0.5, 'map=halos (sm)')
    ax.text(0.05, 0.3, 'model=FastPM+NN')
    ax.text(0.05, 0.1, 'data=FastPM+NN')

    ax = axar[1, 0]
    #ax = fig.add_subplot(gs[1, 0])
    ax.plot(ks, ps1['power'], label='model')
    ax.plot(ks, ps2['power'], label='data', ls = "--")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("initial")
    ax.legend()

    ax = axar[1, 1]
    #ax = fig.add_subplot(gs[1, 1])
    ax.plot(kd, pd1['power'], label='model')
    ax.plot(kd, pd2['power'], label='data', ls = "--")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("final")
    ax.legend()

    ax = axar[1, 2]
    #ax = fig.add_subplot(gs[1, 2])
    ax.plot(km, pm1['power'], label='model')
    ax.plot(km, pm2['power'], label='data', ls = "--")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("map")
    ax.legend()

    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar





def summary(report, figin=None, axin=None, c='k', ls='-', lw=1, alpha=1, label=None, grid=True, filename=None, minor=True,
             lfsize=10, ylsize=14, titsize=14, lncol=2, titles=['Initial', 'Final', 'Model'], tfdensity=True):


    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2 = report

    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
    xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
    xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

    if tfdensity:
        tm = (pm1['power'] / pm2['power']) **0.5
        ts = (ps1['power'] / ps2['power']) **0.5
        td = (pd1['power'] / pd2['power']) **0.5
    else:
        tm = (pm1['power'] / pm2['power']) **1
        ts = (ps1['power'] / ps2['power']) **1
        td = (pd1['power'] / pd2['power']) **1

    from cosmo4d.iotools import create_figure
    if figin is None:
        fig, axar = plt.subplots(2, 3, figsize = (15, 8))
    else:
        fig, axar = figin, axin
    
    ptup = ((ks, xs, ts), (kd, xd, td), (km, xm, tm))

    for i in range(2):
        for j in range(3):
            ax, tup = axar[i, j], ptup[j]
            ax.plot(tup[0], tup[i+1], color=c, ls=ls, lw=lw, label=label, alpha=alpha)
            if grid:
                if minor: wticks = 'minor'
                else: wticks = 'major'
                ax.yaxis.grid(color='gray', linewidth = 0.2, which='major')
                ax.xaxis.grid(color='gray', linewidth = 0.2, which=wticks)
            ax.set_xscale('log')

    for j in range(3):
        axar[0, j].set_title(titles[j], fontsize = titsize)
        axar[0, j].set_ylim(-0.1, 1.2)
        axar[1, j].set_ylim(0, 1.5)
        axar[1, j].set_xlabel('k (h/Mpc)', fontsize = ylsize)

    axar[0, 0].set_ylabel('Cross Correlation', fontsize = ylsize)
    axar[1, 0].set_ylabel('Transfer Function', fontsize = ylsize)
    #legends
    axar[0, 2].legend(loc=0, ncol=lncol, fontsize = lfsize)
    axis = axar[0, 2]
    legends = axis.get_legend_handles_labels()
    nl = len(legends[0])
    axar[0, 1].legend(legends[0][:nl//2], legends[1][:nl//2], fontsize=lfsize)
    axar[0, 2].legend(legends[0][nl//2:], legends[1][nl//2:], fontsize=lfsize)

    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar



def summary4(report, figin=None, axin=None, c='k', ls='-', lw=1, alpha=1, label=None, grid=True, filename=None, minor=True, 
             lfsize=10, ylsize=14, titsize=14, lncol=2, titles=['Initial', 'Model'], tfdensity=True):

    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2 = report

    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
    xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
    xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

    if tfdensity:
        tm = (pm1['power'] / pm2['power']) **0.5
        ts = (ps1['power'] / ps2['power']) **0.5
        td = (pd1['power'] / pd2['power']) **0.5
    else:
        tm = (pm1['power'] / pm2['power']) **1
        ts = (ps1['power'] / ps2['power']) **1
        td = (pd1['power'] / pd2['power']) **1

    from cosmo4d.iotools import create_figure
    if figin is None:
        fig, axar = plt.subplots(2, 2, figsize = (12, 8))
    else:
        fig, axar = figin, axin
    
    ptup = ((ks, xs, ts),  (km, xm, tm))

    for i in range(2):
        for j in range(2):
            ax, tup = axar[i, j], ptup[j]
            ax.plot(tup[0], tup[i+1], color=c, ls=ls, lw=lw, label=label, alpha=alpha)
            if grid:
                if minor: wticks = 'minor'
                else: wticks = 'major'
                ax.yaxis.grid(color='gray', linewidth = 0.2, which='major')
                ax.xaxis.grid(color='gray', linewidth = 0.2, which=wticks)
            ax.set_xscale('log')

    for j in range(2):
        axar[0, j].set_title(titles[j], fontsize = titsize)
        axar[0, j].set_ylim(-0.1, 1.2)
        axar[1, j].set_ylim(0, 1.5)
        axar[1, j].set_xlabel('k (h/Mpc)', fontsize = ylsize)

    axar[0, 0].set_ylabel('Cross Correlation', fontsize = ylsize)
    axar[1, 0].set_ylabel('Transfer Function', fontsize = ylsize)
    #legends
    axar[0, 1].legend(loc=0, ncol=lncol, fontsize = lfsize)
    axis = axar[0, 1]
    legends = axis.get_legend_handles_labels()
    nl = len(legends[0])
    axar[0, 0].legend(legends[0][:nl//2], legends[1][:nl//2], fontsize=lfsize)
    axar[0, 1].legend(legends[0][nl//2:], legends[1][nl//2:], fontsize=lfsize)

    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar




def summary1(report, figin=None, axin=None, c='k', ls='-', lw=1, alpha=1, label=None, lfsize=10, grid=True, filename=None, ylsize=14, titsize=14, 
             minor=True, layout='h', fkey='initial', tfdensity= True, lncol=2, title='Initial', splitlab=True, movelab=False):
    '''Plot tf and rcc for only one of the 3 fields
    '''
    try:
        px, p1, p2 = report
    except:
        if fkey is 'initial' or fkey is 's':
            print('For initial field')
            px, p1, p2 = report[1], report[5], report[6]
        elif fkey is 'final' or fkey is 'd':
            print('For final field')
            px, p1, p2 = report[2], report[7], report[8]
        elif fkey is 'mapp':
            print('For mapp field')
            px, p1, p2 = report[0], report[3], report[4]

    k = px['k']

    cc = px['power'] / (p1['power'] * p2['power']) ** 0.5

    if tfdensity: tt = (p1['power'] / p2['power']) **0.5
    else: tt = (p1['power'] / p2['power'])

    from cosmo4d.iotools import create_figure
    if figin is None:
        if layout is 'v':
            fig, axar = plt.subplots(2, 1, figsize = (5, 8))
        elif layout is 'h':
            fig, axar = plt.subplots(1, 2, figsize = (9, 4))
        else:
            print('layout not understood, should be v or h')
    else:
        fig, axar = figin, axin
    
    axar[0].plot(k, cc, color=c, ls=ls, lw=lw, label=label)
    axar[1].plot(k, tt, color=c, ls=ls, lw=lw, label=label)
    for ax in axar:
        if grid:
            if minor: wticks = 'minor'
        else: wticks = 'major'
        ax.yaxis.grid(color='gray', linewidth = 0.2, which='major')
        ax.xaxis.grid(color='gray', linewidth = 0.2, which=wticks)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=titsize)


    axar[0].set_ylim(-0.1, 1.2)
    axar[1].set_ylim(0, 1.5)
    axar[1].set_xlabel('k (h/Mpc)', fontsize = ylsize)
    if layout =='h': axar[0].set_xlabel('k (h/Mpc)', fontsize = ylsize)

    axar[0].set_ylabel('Cross Correlation', fontsize = 14)
    axar[1].set_ylabel('Transfer Function', fontsize = 14)
    #legends
    axar[1].legend(loc=0, ncol=lncol, fontsize = lfsize)
    if splitlab:
        axis = axar[-1]
        legends = axis.get_legend_handles_labels()
        nl = len(legends[0])
        axar[0].legend(legends[0][:nl//2], legends[1][:nl//2], fontsize=lfsize)
        axar[1].legend(legends[0][nl//2:], legends[1][nl//2:], fontsize=lfsize)
    elif movelab:
        axis = axar[-1]
        legends = axis.get_legend_handles_labels()
        axar[0].legend(legends[0][:], legends[1][:], fontsize=lfsize)
        axar[1].legend().set_visible(False)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar






def evaluate2d(model, data, M0=0, kmin=None, dk=None, Nmu=5, retmesh=False, los=[0, 0, 1]):
    '''return position order:
    xm.power, xs.power, xd.power, 
    pm1.power, pm2.power, 
    ps1.power, ps2.power, 
    pd1.power, pd2.power, 
    data_preview, model_preview
    '''
    from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower

    if kmin is None: kmin = 0 
    if dk is None: dk = 2*numpy.pi/model.s.BoxSize[0]

    model.mapp += M0
    data.mapp += M0

    if abs(model.mapp.cmean()) > 1e-3: modmappmean = model.mapp.cmean()
    else: modmappmean = 1.
    if abs(data.mapp.cmean()) > 1e-3: datmappmean = data.mapp.cmean()
    else: datmappmean = 1.
    modmappmean, datmappmean = 1., 1.
        
    xm = FFTPower(first=FieldMesh(model.mapp/modmappmean), 
                  second=FieldMesh(data.mapp/datmappmean), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
    xd = FFTPower(first=FieldMesh(model.d), second=FieldMesh(data.d), mode='2d', kmin=kmin, dk=dk, 
                  Nmu=Nmu, los=[0, 0, 1])
    xs = FFTPower(first=FieldMesh(model.s), second=FieldMesh(data.s), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, 
                  los=[0, 0, 1])

    pm1 = FFTPower(first=FieldMesh(model.mapp/modmappmean), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
    pd1 = FFTPower(first=FieldMesh(model.d), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
    ps1 = FFTPower(first=FieldMesh(model.s), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])

    pm2 = FFTPower(first=FieldMesh(data.mapp/datmappmean), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
    pd2 = FFTPower(first=FieldMesh(data.d), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
    ps2 = FFTPower(first=FieldMesh(data.s), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])

    if retmesh:
        data_preview = dict(s=[], d=[], mapp=[])
        model_preview = dict(s=[], d=[], mapp=[])

        for axes in [[1, 2], [0, 2], [0, 1]]:
            data_preview['d'].append(data.d.preview(axes=axes))
            data_preview['s'].append(data.s.preview(axes=axes))
            data_preview['mapp'].append(data.mapp.preview(axes=axes))
            model_preview['d'].append(model.d.preview(axes=axes))
            model_preview['s'].append(model.s.preview(axes=axes))
            model_preview['mapp'].append(model.mapp.preview(axes=axes))
        return xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power,\
            data_preview, model_preview
    else:
        return xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power

    
    

def evaluate2d1(model, data, field, M0=0, kmin=None, dk=None, Nmu=5, retmesh=False, los=[0, 0, 1]):
    '''return position order:
    xm.power, xs.power, xd.power, 
    pm1.power, pm2.power, 
    ps1.power, ps2.power, 
    pd1.power, pd2.power, 
    data_preview, model_preview
    '''
    from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower

    if kmin is None: kmin = 0 
    if dk is None: dk = 2*numpy.pi/model.s.BoxSize[0]

    model.mapp += M0
    data.mapp += M0

    if abs(model.mapp.cmean()) > 1e-3: modmappmean = model.mapp.cmean()
    else: modmappmean = 1.
    if abs(data.mapp.cmean()) > 1e-3: datmappmean = data.mapp.cmean()
    else: datmappmean = 1.
    modmappmean, datmappmean = 1., 1.
        
    if field == 'mapp':
        x = FFTPower(first=FieldMesh(model.mapp/modmappmean), 
                  second=FieldMesh(data.mapp/datmappmean), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
        p1 = FFTPower(first=FieldMesh(model.mapp/modmappmean), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
        p2 = FFTPower(first=FieldMesh(data.mapp/datmappmean), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])

    elif field == 's':
        x = FFTPower(first=FieldMesh(model.s), second=FieldMesh(data.s), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
        p1 = FFTPower(first=FieldMesh(model.s), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
        p2 = FFTPower(first=FieldMesh(data.s), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])

    elif field == 'd':
        x = FFTPower(first=FieldMesh(model.d), second=FieldMesh(data.d), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
        p1 = FFTPower(first=FieldMesh(model.d), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
        p2 = FFTPower(first=FieldMesh(data.d), mode='2d', kmin=kmin, dk=dk, Nmu=Nmu, los=[0, 0, 1])
    else:
        print('Field not recongnized')
        return 0

    return x.power, p1.power, p2.power


def summary2dmu(report, figin=None, axin=None, c='k', ls='-', lw=1, alpha=1, label=None, grid=True, filename=None, minor=True,
                lfsize=10, ylsize=14, titsize=14, lncol=2, titles=['Initial', 'Final', 'Model'], muplot=[0], mulab=False, 
                lsmu=['-', '--', ':', '-.'], lsmustyle=True, tfdensity=True):


    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2 = report
    mus = xs.coords['mu']
    Nmu = mus.size

    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
    xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
    xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

    if tfdensity:
        tm = (pm1['power'] / pm2['power']) **0.5
        ts = (ps1['power'] / ps2['power']) **0.5
        td = (pd1['power'] / pd2['power']) **0.5
    else:
        tm = (pm1['power'] / pm2['power']) **1
        ts = (ps1['power'] / ps2['power']) **1
        td = (pd1['power'] / pd2['power']) **1


    from cosmo4d.iotools import create_figure
    if figin is None:
        fig, axar = plt.subplots(2, 3, figsize = (15, 8))
    else:
        fig, axar = figin, axin
    
    ptup = ((ks, xs, ts), (kd, xd, td), (km, xm, tm))
    
    addmulab = True
    for i in range(2):
        for j in range(3):

            ax, tup = axar[i, j], ptup[j]

            for nmu, imu in enumerate(muplot):
                if lsmustyle: ls = lsmu[nmu%len(lsmu)]
                else: ls = ls 
                if mulab:
                    if addmulab and nmu == 0: 
                        labelp = label + '\n$\mu$ = %0.2f'%mus[imu]
                    else: labelp = '$\mu$ = %0.2f'%mus[imu]
                    ax.plot(tup[0][:, imu], tup[i+1][:, imu], ls=ls, lw=lw, label=labelp, alpha=alpha, color=c)
                else: 
                    if not nmu: ax.plot(tup[0][:, imu], tup[i+1][:, imu], ls=ls, lw=lw, label=label, alpha=alpha, color=c)
                    else: ax.plot(tup[0][:, imu], tup[i+1][:, imu], ls=ls, lw=lw, alpha=alpha, color=c)

            if grid:
                if minor: wticks = 'minor'
            else: wticks = 'major'
            ax.yaxis.grid(color='gray', linewidth = 0.2, which='major')
            ax.xaxis.grid(color='gray', linewidth = 0.2, which=wticks)
            ax.set_xscale('log')

    for j in range(3):
        axar[0, j].set_title(titles[j], fontsize = titsize)
        axar[0, j].set_ylim(-0.1, 1.2)
        axar[1, j].set_ylim(0, 1.5)
        axar[1, j].set_xlabel('k (h/Mpc)', fontsize = ylsize)

    axar[0, 0].set_ylabel('Cross Correlation', fontsize = ylsize)
    axar[1, 0].set_ylabel('Transfer Function', fontsize = ylsize)
    #legends
    axar[0, 2].legend(loc=0, ncol=lncol, fontsize = lfsize)
    axis = axar[0, 2]
    legends = axis.get_legend_handles_labels()
    nl = len(legends[0])
    axar[0, 1].legend(legends[0][:nl//2], legends[1][:nl//2], fontsize=lfsize)
    axar[0, 2].legend(legends[0][nl//2:], legends[1][nl//2:], fontsize=lfsize)

    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar



def summary2dmu1(report, figin=None, axin=None, c='k', ls='-', lw=1, alpha=1, label=None, grid=True, filename=None, minor=True,
                lfsize=10, ylsize=14, titsize=14, lncol=2, titles=['Initial', 'Final', 'Model'], muplot=[0], mulab=False, 
                 lsmu=['-', '--', ':', '-.'], lsmustyle=True, fkey='s', layout='h', splitlab=True, movelab=False, tfdensity=True):


    try:
        px, p1, p2 = report
    except:
        if fkey is 'initial' or fkey is 's':
            print('For initial field')
            px, p1, p2 = report[1], report[5], report[6]
        elif fkey is 'final' or fkey is 'd':
            print('For final field')
            px, p1, p2 = report[2], report[7], report[8]
        elif fkey is 'mapp':
            print('For mapp field')
            px, p1, p2 = report[0], report[3], report[4]

    k = px.coords['k']
    mus = px.coords['mu']
    Nmu = mus.size

    cc = px['power'] / (p1['power'] * p2['power']) ** 0.5

    if tfdensity: tt = (p1['power'] / p2['power']) **0.5
    else: tt = (p1['power'] / p2['power'])

    from cosmo4d.iotools import create_figure
    if figin is None:
        if layout is 'v':
            fig, axar = plt.subplots(2, 1, figsize = (5, 8))
        elif layout is 'h':
            fig, axar = plt.subplots(1, 2, figsize = (9, 4))
        else:
            print('layout not understood, should be v or h')
    else:
        fig, axar = figin, axin
    

    addmulab = True
    for nmu, imu in enumerate(muplot):
        if lsmustyle: ls = lsmu[nmu%len(lsmu)]
        else: ls = ls 
        if mulab:
            if addmulab and nmu == 0: 
                labelp = label + '\n$\mu$ = %0.2f'%mus[imu]
            else: labelp = '$\mu$ = %0.2f'%mus[imu]
            axar[0].plot(k, cc[:, imu], ls=ls, lw=lw, label=labelp, alpha=alpha, color=c)
            axar[1].plot(k, tt[:, imu], ls=ls, lw=lw, label=labelp, alpha=alpha, color=c)
        else: 
            if not nmu:
                axar[0].plot(k, cc[:, imu], ls=ls, lw=lw, label=label, alpha=alpha, color=c)
                axar[1].plot(k, tt[:, imu], ls=ls, lw=lw, label=label, alpha=alpha, color=c)
            else:
                axar[0].plot(k, cc[:, imu], ls=ls, lw=lw, alpha=alpha, color=c)
                axar[1].plot(k, tt[:, imu], ls=ls, lw=lw, alpha=alpha, color=c)

#
    for ax in axar:
        if grid:
            if minor: wticks = 'minor'
        else: wticks = 'major'
        ax.yaxis.grid(color='gray', linewidth = 0.2, which='major')
        ax.xaxis.grid(color='gray', linewidth = 0.2, which=wticks)
        ax.set_xscale('log')


    axar[0].set_ylim(-0.1, 1.2)
    axar[1].set_ylim(0, 1.5)
    axar[1].set_xlabel('k (h/Mpc)', fontsize = ylsize)
    if layout =='h': axar[0].set_xlabel('k (h/Mpc)', fontsize = ylsize)

    axar[0].set_ylabel('Cross Correlation', fontsize = 14)
    axar[1].set_ylabel('Transfer Function', fontsize = 14)
    #legends
    axar[1].legend(loc=0, ncol=lncol, fontsize = lfsize)
    if splitlab:
        axis = axar[-1]
        legends = axis.get_legend_handles_labels()
        nl = len(legends[0])
        axar[0].legend(legends[0][:nl//2], legends[1][:nl//2], fontsize=lfsize)
        axar[1].legend(legends[0][nl//2:], legends[1][nl//2:], fontsize=lfsize)
    elif movelab:
        axis = axar[-1]
        legends = axis.get_legend_handles_labels()
        axar[0].legend(legends[0][:], legends[1][:], fontsize=lfsize)
        axar[1].legend().set_visible(False)


    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar


    

def summary2dall(report, figin=None, axin=None, c='k', ls='-', lw=1, alpha=1, label=None, grid=True, filename=None, minor=True,
               lfsize=10, ylsize=14, titsize=14, lncol=2, titles=['Initial', 'Final', 'Model'], skip=1, mulab=True):


    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2 = report
    mus = xs.coords['mu']
    Nmu = mus.size

    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
    xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
    xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

    tm = (pm1['power'] / pm2['power']) **0.5
    ts = (ps1['power'] / ps2['power']) **0.5
    td = (pd1['power'] / pd2['power']) **0.5

    from cosmo4d.iotools import create_figure
    if figin is None:
        fig, axar = plt.subplots(2, 3, figsize = (15, 8))
    else:
        fig, axar = figin, axin
    
    ptup = ((ks, xs, ts), (kd, xd, td), (km, xm, tm))
    
    for i in range(2):
        for j in range(3):
            ax, tup = axar[i, j], ptup[j]
            for mu in range(0+int(skip/2), Nmu+skip-int(skip/2)-1, skip):
            #for mu in range(Nmu):
                if mulab: ax.plot(tup[0][:, mu], tup[i+1][:, mu], ls=ls, lw=lw, label='$\mu$ = %0.2f'%mus[mu], alpha=alpha)
                else: ax.plot(tup[0][:, mu], tup[i+1][:, mu], ls=ls, lw=lw, alpha=alpha)
            if grid:
                if minor: wticks = 'minor'
            else: wticks = 'major'
            ax.yaxis.grid(color='gray', linewidth = 0.2, which='major')
            ax.xaxis.grid(color='gray', linewidth = 0.2, which=wticks)
            ax.set_xscale('log')

    for j in range(3):
        axar[0, j].set_title(titles[j], fontsize = titsize)
        axar[0, j].set_ylim(-0.1, 1.2)
        axar[1, j].set_ylim(0, 1.5)
        axar[1, j].set_xlabel('k (h/Mpc)', fontsize = ylsize)

    axar[0, 0].set_ylabel('Cross Correlation', fontsize = ylsize)
    axar[1, 0].set_ylabel('Transfer Function', fontsize = ylsize)
    #legends
    axar[0, 2].legend(loc=0, ncol=lncol, fontsize = lfsize)
    axis = axar[0, 2]
    legends = axis.get_legend_handles_labels()
    nl = len(legends[0])
    axar[0, 1].legend(legends[0][:nl//2], legends[1][:nl//2], fontsize=lfsize)
    axar[0, 2].legend(legends[0][nl//2:], legends[1][nl//2:], fontsize=lfsize)

    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar



##def smobj(pm, mock_model, noise_model, data, prior_ps, sml, noised = 2, smooth=None, M0=1e8, L1=False, offset=False, smoothprior=False, ftw=False, mock_models=None, offar=None, ivarar=None):
##
##    try:
##        model = mock_model.mapp
##        dlineark = mock_model.s.r2c()
##    except:
##        model = mock_model
##        if mock_models is not None: dlineark = mock_models.r2c()
##        else: 
##            print('Need initial field for prior')
##            return None
##
##    try:
##        data = data.mapp
##    except:
##        data = data
##
##    #sm
##    if smooth is not None:
##        def fingauss(pm, R):
##            kny = numpy.pi*pm.Nmesh[0]/pm.BoxSize[0]
##            def tf(k):
##                k2 = sum(((2*kny/numpy.pi)*numpy.sin(ki*numpy.pi/(2*kny)))**2  for ki in k)
##                wts = numpy.exp(-0.5*k2* R**2)
##                return wts
##            return tf            
##        tf = fingauss(pm, smooth) 
##        data = data.r2c().apply(lambda k, v: tf(k )*v).c2r()
##        model = model.r2c().apply(lambda k, v: tf(k )*v).c2r()
##
##    logdataM0 = pm.create(mode = 'real')
##    logdataM0.value[...] = numpy.log(data + M0)
##    logmodelM0 = pm.create(mode = 'real')
##    logmodelM0.value[...] = numpy.log(model + M0)
##    residual = (logmodelM0 - logdataM0)
##
##    if offset:
##        #offset array has data-model, so added to residual i.e. model-data
##        try:
##            residual += noise_model.offset
##        except:
##            residual += offar
##
##    if noised == 2:
##        if pm.comm.rank == 0:
##            print('2D noise model')
##        try:
##            residual *= noise_model.ivar2d ** 0.5
##        except:
##            residual *= ivarar
##    elif noised == 3:
##        if pm.comm.rank == 0:
##            print('3D noise model')
##        try:
##            residual *= noise_model.ivar3d ** 0.5
##        except:
##            residual *= ivarar
##
##    if ftw:
##        residual *= noise_model.ivar2d ** 0.5
##
##    #Smooth
##    smooth_window = lambda k: numpy.exp(- sml ** 2 * sum(ki ** 2 for ki in k))
##
##    residual = residual.r2c().apply(lambda k, v: smooth_window(k )*v).c2r()
##
#####################
##    #LNorm
##    if L1:
##        if pm.comm.rank == 0:
##            print('L1 norm objective is not defined\n\n')
##        return None
##    else:
##        if pm.comm.rank == 0:
##            print('L2 norm objective')
##        residual = residual.cnorm()
##
##    #Prior
##    def tfps(k):
##        k2 = sum(ki**2 for ki in k)
##        r = (prior_ps(k2 ** 0.5) / pm.BoxSize.prod()) ** -0.5
##        r[k2 == 0] = 1.0
##        return r
##    prior = dlineark.apply(lambda k, v: tfps(k )*v).c2r()
##    prior = prior.cnorm()*pm.Nmesh.prod()**-1.
##
##    return residual+prior, residual, prior





#
#def loadfile(key, folder, ipath, container, subf = 'best-fit', mesh=False, keycheck=True, keyskip = True, verbose=False, kmin=None, dk=None):
#    report, reportf, bestp, datap, fitp = container
#    if key in report.keys() and keycheck:
#        if verbose: print('Key clash for %s'%key)
#        if keyskip: return None
#    bestm = mapp.Observable.load(ipath + '%s/%s'%(folder, subf))
#    datam = mapp.Observable.load(ipath + 'datap')
#    fitm = mapp.Observable.load(ipath + 'fitp').mapp
#    report[key] = evaluate(bestm, datam, kmin=kmin, dk=dk)[:-2]
#    reportf[key] = evaluate1(fitm, datam.mapp, kmin=kmin, dk=dk)
#    if mesh:
#        bestp[key], datap[key], fitp[key] = bestm, datam, fitm
#


def loadfile(key, folder, ipath, reports, subf = 'best-fit', mesh=False, keycheck=True, keyskip = True, verbose=False, kmin=None, dk=None):
    if key in reports.keys() and keycheck:
        if verbose: print('Key clash for %s'%key)
        if keyskip: return None
    bestm = mapp.Observable.load(ipath + '%s/%s'%(folder, subf))
    datam = mapp.Observable.load(ipath + 'datap')
    fitm = mapp.Observable.load(ipath + 'fitp').mapp
    report = evaluate(bestm, datam, kmin=kmin, dk=dk)[:-2]
    reportf = evaluate1(fitm, datam.mapp, kmin=kmin, dk=dk)
    if mesh:
        bestp, datap, fitp = bestm, datam, fitm
    else:
        bestp, datap, fitp = None, None, None
    return key, report, reportf, bestp, datap, fitp
        

#def loadfile(key, folder, ipath, subf = 'best-fit', mesh=False, keycheck=True, keyskip = True, verbose=False, kmin=None, dk=None):
#    if key in report.keys() and keycheck:
#        if verbose: print('Key clash for %s'%key)
#        if keyskip: return None
#    bestm = mapp.Observable.load(ipath + '%s/%s'%(folder, subf))
#    datam = mapp.Observable.load(ipath + 'datap')
#    fitm = mapp.Observable.load(ipath + 'fitp').mapp
#    report[key] = evaluate(bestm, datam, kmin=kmin, dk=dk)[:-2]
#    reportf[key] = evaluate1(fitm, datam.mapp, kmin=kmin, dk=dk)
#    if mesh:
#        bestp[key], datap[key], fitp[key] = bestm, datam, fitm
#        


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d




        
def rpobject(rep):
    '''Convert report from dict to objects
    '''
    pskey = ['xm', 'xs', 'xd', 'pmr', 'pmd', 'psr', 'psd', 'pdr', 'pdd']
    repdict = {}
    for key in rep.keys():
        toret = {}
        for i, j in enumerate(pskey):
            toret[j] = rep[key][i]['power']
        toret['k'] = rep[key][i]['k']
        toret['modes'] = rep[key][i]['modes']
        repdict[key] = objectview(toret)
    return repdict

def skeys(pattern, dd):
    '''Return keys in dict=dd which have pattern=pattern
    '''
    toret = []
    for key in dd.keys():
        if pattern in key: toret.append(key)
    return toret


def evalrep(report, field='s', returnp=False):
    '''Retrun rcc and tf given a report and field = s, d, or mapp
    '''
    if field == 's': x, r, d = 1, 5, 6
    elif field == 'd': x, r, d = 2, 7, 8
    elif field == 'mapp': x, r, d = 0, 3, 4
    rcc = report[x]['power'] / (report[r]['power']*report[d]['power'])**0.5
    tf = (report[r]['power']/report[d]['power'])**0.5
    if returnp: return rcc, tf, [report[x]['power'], report[r]['power'], report[d]['power']]
    else: return rcc, tf
















def summarystoch(report, figin=None, axin=None, c='k', ls='-', lw=1, alpha=1, label=None, grid=True, filename=None, minor=True,
             lfsize=10, ylsize=14, titsize=14, lncol=2, titles=['Initial', 'Final', 'Model']):


    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2, psd, pdd, pmd = report

    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
    xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
    xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

    tm = (pm1['power'] / pm2['power']) **0.5
    ts = (ps1['power'] / ps2['power']) **0.5
    td = (pd1['power'] / pd2['power']) **0.5

    ss = ps2['power']*(1 - xs**2)
    sd = pd2['power']*(1 - xd**2)
    sm = pm2['power']*(1 - xm**2)

    from cosmo4d.iotools import create_figure
    if figin is None:
        fig, axar = plt.subplots(4, 3, figsize = (15, 15))
    else:
        fig, axar = figin, axin
    
    #ptup = ((ks, xs, ts, psd['power']), (kd, xd, td, pdd['power']), (km, xm, tm, pmd['power']))
    ptup = ((ks, xs, ts, psd['power'], ss), (kd, xd, td, pdd['power'], sd), (km, xm, tm, pmd['power'], sm))

    for i in range(4):
        for j in range(3):
            ax, tup = axar[i, j], ptup[j]
            ax.plot(tup[0], tup[i+1], color=c, ls=ls, lw=lw, label=label, alpha=alpha)
            if grid:
                if minor: wticks = 'minor'
                else: wticks = 'major'
                ax.yaxis.grid(color='gray', linewidth = 0.2, which='major')
                ax.xaxis.grid(color='gray', linewidth = 0.2, which=wticks)
            ax.set_xscale('log')
            
    for j, p in enumerate([ps2['power'], pd2['power'], pm2['power']]):
        ax = axar[2, j]
        ax.plot(ks, p, color=c, ls="--", lw=2, label=label, alpha=0.5)
    
    for j in range(3):
        axar[0, j].set_title(titles[j], fontsize = titsize)
        axar[0, j].set_ylim(-0.1, 1.2)
        axar[1, j].set_ylim(0, 1.5)
        axar[1, j].set_xlabel('k (h/Mpc)', fontsize = ylsize)

    axar[0, 0].set_ylabel('Cross Correlation', fontsize = ylsize)
    axar[1, 0].set_ylabel('Transfer Function', fontsize = ylsize)
    #legends
    axar[0, 2].legend(loc=0, ncol=lncol, fontsize = lfsize)
    axis = axar[0, 2]
    legends = axis.get_legend_handles_labels()
    nl = len(legends[0])
    axar[0, 1].legend(legends[0][:nl//2], legends[1][:nl//2], fontsize=lfsize)
    axar[0, 2].legend(legends[0][nl//2:], legends[1][nl//2:], fontsize=lfsize)


    if filename is not None:
        fig.savefig(filename)

    if figin is None:
        return fig, axar

