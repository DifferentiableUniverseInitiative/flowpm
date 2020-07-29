#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import warnings

def evaluate(model, data):
    from nbodykit.lab import FieldMesh, FFTPower, ProjectedFFTPower

    #print('Means are : ', model.mapp.cmean(), data.mapp.cmean())
    if abs(model.mapp.cmean()) > 1e-3: modmappmean = model.mapp.cmean()
    else: modmappmean = 1.
    if abs(data.mapp.cmean()) > 1e-3: datmappmean = data.mapp.cmean()
    else: datmappmean = 1.
    modmappmean, datmappmean = 1., 1.
        
    xm = FFTPower(first=FieldMesh(model.mapp/modmappmean), second=FieldMesh(data.mapp/datmappmean), mode='1d')
    xd = FFTPower(first=FieldMesh(model.d), second=FieldMesh(data.d), mode='1d')
    xs = FFTPower(first=FieldMesh(model.s), second=FieldMesh(data.s), mode='1d')

    pm1 = FFTPower(first=FieldMesh(model.mapp/modmappmean), mode='1d')
    pd1 = FFTPower(first=FieldMesh(model.d), mode='1d')
    ps1 = FFTPower(first=FieldMesh(model.s), mode='1d')

    pm2 = FFTPower(first=FieldMesh(data.mapp/datmappmean), mode='1d')
    pd2 = FFTPower(first=FieldMesh(data.d), mode='1d')
    ps2 = FFTPower(first=FieldMesh(data.s), mode='1d')

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

    return xm.power, xs.power, xd.power, pm1.power, pm2.power, ps1.power, ps2.power, pd1.power, pd2.power, data_preview, model_preview

def save_report(report, filename, pm, title=None):
    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2, data_preview, model_preview = report

    from cosmo4d.iotools import create_figure
    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
        xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
        xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

        tm = (pm1['power'] / pm2['power']) **0.5
        ts = (ps1['power'] / ps2['power']) **0.5
        td = (pd1['power'] / pd2['power']) **0.5

        fig, gs = create_figure((12, 9), (4, 6))
        for i in range(3):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(data_preview['s'][i])
            ax.set_title("s data")

        for i in range(3):
            ax = fig.add_subplot(gs[0, i + 3])
            ax.imshow(data_preview['d'][i])
            ax.set_title("d data")

        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(model_preview['s'][i])
            ax.set_title("s model")

        for i in range(3):
            ax = fig.add_subplot(gs[1, i + 3])
            ax.imshow(model_preview['d'][i])
            ax.set_title("d model")

        for i in range(3):
            ax = fig.add_subplot(gs[2, i + 3])
            ax.imshow(data_preview['mapp'][i])
            ax.set_title("map data")

        for i in range(3):
            ax = fig.add_subplot(gs[3, i + 3])
            ax.imshow(model_preview['mapp'][i])
            ax.set_title("map model")

        ax = fig.add_subplot(gs[2, 0])
        ax.plot(ks, xs, label='intial', ls = "-", lw=1.5)
        ax.plot(kd, xd, label='final', ls = "--", lw=1.5)
        ax.plot(km, xm, label='map', ls = ":", lw=2)
        ax.legend()
        ax.set_xscale('log')
        ax.set_title('Cross coeff')

        ax = fig.add_subplot(gs[2, 1])
        ax.plot(ks, ts, label='intial', ls = "-", lw=1.5)
        ax.plot(kd, td, label='final', ls = "--", lw=1.5)
        ax.plot(km, tm, label='map', ls = ":", lw=2)
        ax.legend()
        ax.set_xscale('log')
        ax.set_title('Transfer func')

        ax = fig.add_subplot(gs[2, 2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.05, 0.9, 's=linear')
        ax.text(0.05, 0.7, 'd=non-linear')
        ax.text(0.05, 0.5, 'map=halos (sm)')
        ax.text(0.05, 0.3, 'model=FastPM+NN')
        ax.text(0.05, 0.1, 'data=FastPM+NN')

        ax = fig.add_subplot(gs[3, 0])
        ax.plot(ks, ps1['power'], label='model')
        ax.plot(ks, ps2['power'], label='data', ls = "--")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("initial")
        ax.legend()

        ax = fig.add_subplot(gs[3, 1])
        ax.plot(kd, pd1['power'], label='model')
        ax.plot(kd, pd2['power'], label='data', ls = "--")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("final")
        ax.legend()

        ax = fig.add_subplot(gs[3, 2])
        ax.plot(km, pm1['power'], label='model')
        ax.plot(km, pm2['power'], label='data', ls = "--")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("map")
        ax.legend()

    if title is not None: fig.suptitle(title)
    fig.tight_layout()
    if pm.comm.rank == 0:
        fig.savefig(filename)



def save_2ptreport(report, filename, pm, title=None):
    xm, xs, xd, pm1, pm2, ps1, ps2, pd1, pd2, data_preview, model_preview = report

    from cosmo4d.iotools import create_figure

    km = xm['k']
    ks = xs['k']
    kd = xd['k']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xm = xm['power'] / (pm1['power'] * pm2['power']) ** 0.5
        xs = xs['power'] / (ps1['power'] * ps2['power']) ** 0.5
        xd = xd['power'] / (pd1['power'] * pd2['power']) ** 0.5

        tm = (pm1['power'] / pm2['power']) **0.5
        ts = (ps1['power'] / ps2['power']) **0.5
        td = (pd1['power'] / pd2['power']) **0.5

        fig, gs = create_figure((10, 6), (2, 3))
        #fig, axar = plt.subplots(2, 3, figsize = (15, 8))

        #ax = axar[0, 0]
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(ks, xs, label='intial', ls = "-", lw=1.5)
        ax.plot(kd, xd, label='final', ls = "--", lw=1.5)
        ax.plot(km, xm, label='map', ls = ":", lw=2)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.minorticks_on()
        ax.set_xscale('log')
        ax.grid(which='both', lw=0.3, color='gray')
        ax.set_title('Cross coeff')

        #ax = axar[0, 1]
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(ks, ts, label='intial', ls = "-", lw=1.5)
        ax.plot(kd, td, label='final', ls = "--", lw=1.5)
        ax.plot(km, tm, label='map', ls = ":", lw=2)
        ax.legend()
        ax.set_xscale('log')
        ax.grid(which='both', lw=0.3, color='gray')
        ax.set_title('Transfer func')

    #    ax = axar[0, 2]
        ax = fig.add_subplot(gs[0, 2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.05, 0.9, 's=linear')
        ax.text(0.05, 0.7, 'd=non-linear')
        ax.text(0.05, 0.5, 'map=halos (sm)')
        ax.text(0.05, 0.3, 'model=FastPM+NN')
        ax.text(0.05, 0.1, 'data=FastPM+NN')
        ax.grid(which='both', lw=0.3, color='gray')
        if title is not None: ax.set_title(title)

    #    ax = axar[1, 0]
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(ks, ps1['power'], label='model')
        ax.plot(ks, ps2['power'], label='data', ls = "--")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("initial")
        ax.legend()
        ax.grid(which='both', lw=0.3, color='gray')

        #ax = axar[1, 1]
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(kd, pd1['power'], label='model')
        ax.plot(kd, pd2['power'], label='data', ls = "--")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("final")
        ax.legend()
        ax.grid(which='both', lw=0.3, color='gray')

        #ax = axar[1, 2]
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(km, pm1['power'], label='model')
        ax.plot(km, pm2['power'], label='data', ls = "--")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("map")
        ax.legend()
        ax.grid(which='both', lw=0.3, color='gray')

    #if title is not None: fig.suptitle(title)
    fig.tight_layout()
    if pm.comm.rank == 0:
        fig.savefig(filename)



