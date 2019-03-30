import numpy as np
import numpy
from nbodykit.lab import BigFileMesh, BigFileCatalog
from pmesh.pm import ParticleMesh

from nbodykit.cosmology import Cosmology, EHPower, Planck15
from background import *


def laplace(k, v):
    kk = sum(ki ** 2 for ki in k)
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    b = v / kk
    b[mask] = 0
    return b



def gradient(dir, order=1):
    if order == 0:
        def kernel(k, v):
            # clear the nyquist to ensure field is real
            mask = v.i[dir] != v.Nmesh[dir] // 2
            return v * (1j * k[dir]) * mask
    if order == 1:
        def kernel(k, v):
            cellsize = (v.BoxSize[dir] / v.Nmesh[dir])
            w = k[dir] * cellsize

            a = 1 / (6.0 * cellsize) * (8 * numpy.sin(w) - numpy.sin(2 * w))
            # a is already zero at the nyquist to ensure field is real
            return v * (1j * a)
    return kernel






def fknlongrange(r_split):
    if r_split != 0:
        def kernel(k, v):
            kk = sum(ki ** 2 for ki in k)
            return v * numpy.exp(-kk * r_split**2)
    else:
        def kernel(k, v):
            return v
    return kernel


def longrange(x, delta_k, split, factor):
    """ factor shall be 3 * Omega_M / 2, if delta_k is really 1 + overdensity """

    return longrange_batch([x], delta_k, split, factor)[0]


def longrange_batch(x, delta_k, split, factor):
    """ like long range, but x is a list of positions """
    # use the four point kernel to suppresse artificial growth of noise like terms

    f = [numpy.empty_like(xi) for xi in x]

    pot_k = delta_k.apply(laplace) \
                  .apply(fknlongrange(split), out=Ellipsis)

    for d in range(delta_k.ndim):
        force_d = pot_k.apply(gradient(d, order=1)) \
                  .c2r(out=Ellipsis)
        for xi, fi in zip(x, f):
            force_d.readout(xi, out=fi[..., d])

    for fi in f:
        fi[...] *= factor

    return f


