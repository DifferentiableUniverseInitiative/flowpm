from __future__ import print_function
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif

import numpy

from pmesh.abopt import ParticleMesh, RealField, ComplexField, check_grad

from nbodykit.cosmology import Planck15, EHPower
cosmo = Planck15.clone(Tcmb0=0)

pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4, 4), dtype='f8')
pk = EHPower(Planck15, redshift=0)

mask2d = pm.resize([4, 4, 1]).create(mode='real')
mask2d[...] = 1.0

from cosmo4d import map2d
from cosmo4d.nbody import NBodyModel
from cosmo4d.options import UseComplexSpaceOptimizer, UseRealSpaceOptimizer

def test_map2d():
    dynamic_model = NBodyModel(cosmo, pm, 1, [1.0])
    mock_model = map2d.MockModel(dynamic_model)
    noise_model = map2d.NoiseModel(pm, mask2d, 1.0, 1234)

    initial = pm.generate_whitenoise(1234, mode='real')

    obs = mock_model.make_observable(initial)
    assert_array_equal(obs.map2d.Nmesh , (4, 4, 1))
    obsn = noise_model.add_noise(obs)
    assert_array_equal(obsn.map2d.Nmesh , (4, 4, 1))

    print((obsn.map2d - obs.map2d).cshape)

    obj = map2d.Objective(mock_model, noise_model, obsn, prior_ps=pk)
    prob = obj.get_problem(atol=0.01, precond=UseComplexSpaceOptimizer)
    print(prob.f(initial))

    obsn.save("obsn")
    obsn2 = map2d.Observable.load("obsn")

    assert_array_equal(obsn2.map2d, obsn.map2d)

