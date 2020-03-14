import numpy as np
import matplotlib.pyplot as plt
import tools


def saveimfig(i, iterand, truth, fpath):

    ic, fin = truth
    ic1, fin1 = iterand

    plt.figure(figsize=(15,3))
    plt.subplot(141)
    plt.imshow(ic[0].sum(axis=2))
    plt.title('Intial')
    plt.colorbar()

    plt.subplot(142)
    plt.imshow(fin[0].sum(axis=2))
    plt.title('Final') 
    plt.colorbar()

    plt.subplot(143)
    plt.imshow(ic1[0].sum(axis=2))
    plt.title('Recon Init')
    plt.colorbar()

    plt.subplot(144)
    plt.imshow((fin1[0]).sum(axis=2))
    plt.title('Recon Final')
    plt.colorbar()

    plt.tight_layout()
    if type(i) == str: plt.savefig(fpath + "/reconim%s.png"%i)
    elif type(i) == int: plt.savefig(fpath + "/reconim%04d.png"%i)
    
    plt.close()


def saveimfig2x2(i, iterand, truth, fpath, cmapic='RdYlBu_r', cmapfin='YlOrRd', vminic=-120, vmaxic=120, vminfin=0, vmaxfin=350):

    ic, fin = truth
    ic1, fin1 = iterand

    plt.figure(figsize=(7, 6))
    plt.subplot(221)
    plt.imshow(ic[0].sum(axis=2), vmin=vminic, vmax=vmaxic, cmap=cmapic)
    plt.title('True Intial')
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(fin[0].sum(axis=2), vmin=vminfin, vmax=vmaxfin, cmap=cmapfin)
    plt.title('True Final') 
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(ic1[0].sum(axis=2), vmin=vminic, vmax=vmaxic, cmap=cmapic)
    plt.title('Reconstruct Initial')
    plt.colorbar()

    plt.subplot(224)
    plt.imshow(fin1[0].sum(axis=2), vmin=vminfin, vmax=vmaxfin, cmap=cmapfin)
    plt.title('Reconstruct Final')
    plt.colorbar()

    plt.tight_layout()
    if type(i) == str: plt.savefig(fpath + "/reconim%s.png"%i)
    elif type(i) == int: plt.savefig(fpath + "/reconim%04d.png"%i)
    plt.close()




def save2ptfig(i, iterand, truth, fpath, bs, fsize=12):

    ic, fin = truth
    ic1, fin1 = iterand

    pks = []
    if ic1[0].mean() < 1e-3: ic1[0] += 1
    if ic[0].mean() < 1e-3: ic[0] += 1
    k, p1 = tools.power(1+ic1[0], boxsize=bs)
    k, p2 = tools.power(1+ic[0], boxsize=bs)
    k, p12 = tools.power(1+ic1[0], f2=1+ic[0], boxsize=bs)
    pks.append([p1, p2, p12])
    if fin1[0].mean() < 1e-3: fin1[0] += 1
    if fin[0].mean() < 1e-3: fin[0] += 1
    k, p1 = tools.power(fin1[0], boxsize=bs)
    k, p2 = tools.power(fin[0], boxsize=bs)
    k, p12 = tools.power(fin1[0], f2=fin[0], boxsize=bs)
    pks.append([p1, p2, p12])
    

    fig, ax = plt.subplots(1, 3, figsize=(12,3.5))
    ax[0].plot(k, pks[0][0], 'C0', lw=2, label='Recon')
    ax[0].plot(k, pks[0][1], 'C0--', lw=2, label='Truth')
    ax[0].plot(k, pks[1][0], 'C1', lw=2)
    ax[0].plot(k, pks[1][1], 'C1--', lw=2)
    ax[0].loglog()
    ax[0].set_ylabel('P(k)')
    
    p1, p2, p12 = pks[0]
    ax[1].plot(k, p12/(p1*p2)**0.5, 'C0', lw=2, label='Init')
    ax[2].plot(k, (p1/p2)**0.5, 'C0', lw=2, label='')
    
    p1, p2, p12 = pks[1]
    ax[1].plot(k, p12/(p1*p2)**0.5, 'C1', lw=2, label='Final')
    ax[2].plot(k, (p1/p2)**0.5, 'C1', lw=2, label='')
               
    ax[1].semilogx()
    ax[2].semilogx()
    ax[1].set_ylim(-0.1, 1.1)
    ax[2].set_ylim(-0.1, 2.)
    ax[1].set_ylabel('Cross correlation', fontsize=fsize)
    ax[2].set_ylabel('Transfer Function', fontsize=fsize)
    
    for axis in ax:
        axis.legend(fontsize=fsize)
        axis.grid(which='both')
        axis.set_xlabel('k (h/Mpc)', fontsize=fsize)
    plt.tight_layout()
    plt.savefig(fpath + '/recon2pt%s.png'%str(i))
    plt.close()



def savefile():
    pass
