#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:48:34 2021

@author: Denise Lanzieri
"""

from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter   
import numpy as np
import lenstools as lt
from lenstools.simulations import DensityPlane
import astropy.units as u
from astropy.cosmology import Planck15
import tensorflow as tf
from flowpm.tfbackground import rad_comoving_distance
import flowpm.constants as constants
from flowpm.tfpower import linear_matter_power
import flowpm
from  flowpm.raytracing import lightcone,convergenceBorn
from flowpm.io import save_state
import tensorflow_addons as tfa
import bigfile
from lenstools.simulations.nbody import NbodySnapshot

#%%
z_source=1.
field=5.
box_size=200.
nc=1024
Omega_c= 0.2589
sigma8= 0.8159
nsteps=11
z=1.0
#%%
z2a=lambda z:1. / (1. + z)
a2z = lambda a: 1/a - 1
#%%
plane_size=64
class FastPMSnapshot(NbodySnapshot):

	"""
	A class that handles FastPM simulation snapshots
	"""

	_header_keys = ['masses','num_particles_file','num_particles_total','box_size','num_files','Om0','Ode0','h']

	############################
	#Open the file with bigfile#
	############################

	@classmethod
	def open(cls,filename,pool=None,header_kwargs=dict(),**kwargs):

		if bigfile is None:
			raise ImportError("bigfile must be installed!")

		fp = bigfile.BigFile(cls.buildFilename(filename,pool,**kwargs))
		return cls(fp,pool,header_kwargs=header_kwargs)

	###################################################################################
	######################Abstract method implementation###############################
	###################################################################################

	@classmethod
	def buildFilename(cls,root,pool):
		return root

	@classmethod
	def int2root(cls,name,n):
		return name

	def getHeader(self):

		#Initialize header
		header = dict()
		bf_header = self.fp["Header"].attrs

		###############################################
		#Translate fastPM header into lenstools header#
		###############################################

		#Number of particles/files
		header["num_particles_file"] = bf_header["NC"][0]**3
		header["num_particles_total"] = header["num_particles_file"]
		header["num_files"] = 1

		#Cosmology
		header["Om0"] = bf_header["OmegaCDM"][0] + bf_header["OmegaB"][0]
		header["Ode0"] = 1. - header["Om0"]
		header["w0"] = -1.
		header["wa"] = 0.
		header["h"] = bf_header["h"][0] ; header["redshift"] = a2z(bf_header["Time"][0]);
		header["comoving_distance"] = bf_header["comoving_distance"][0]*1.0e3
		header["scale_factor"] = bf_header["Time"][0]     

		#Box size in kpc/h
		header["box_size"] = bf_header["BoxSize"][0]*1.0e3
        #Plane Resolution
		#Masses
		header["masses"] = np.array([0.,bf_header["M0"][0]*header["h"],0.,0.,0.,0.])
		#################

		return header

	def setLimits(self):

		if self.pool is None:
			self._first = None
			self._last = None
		else:

			#Divide equally between tasks
			Nt,Np = self.pool.size+1,bigfile.BigData(self.fp).size
			part_per_task = Np//Nt
			self._first = part_per_task*self.pool.rank
			self._last = part_per_task*(self.pool.rank+1)

			#Add the remainder to the last task
			if (Np%Nt) and (self.pool.rank==Nt-1):
				self._last += Np%Nt

	def getPositions(self,first=None,last=None,save=True):

		#Get data pointer
		data = self.fp
		
		#Read in positions in Mpc/h
		if (first is None) or (last is None):
			positions = (data["0/Position"][:] + field/plane_size*box_size )*self.Mpc_over_h
		else:
			positions = data["0/Position"][first:last]*self.Mpc_over_h

		#Enforce periodic boundary conditions
		for n in (0,1):
			positions[:,n][positions[:,n]<0] += self.header["box_size"]
			positions[:,n][positions[:,n]>self.header["box_size"]] -= self.header["box_size"]

		#Maybe save
		if save:
			self.positions = positions

		#Initialize useless attributes to None
		self.weights = None
		self.virial_radius = None
		self.concentration = None

		#Return
		return positions 
###########################################################################################

	def getVelocities(self,first=None,last=None,save=True):
		raise NotImplementedError

	def getID(self,first=None,last=None,save=True):
		raise NotImplementedError

	def write(self,filename,files=1):
		raise NotImplementedError


#%%
def lenstools_raytracer(plane_size,box_size,field):
# Create a tracer object    
    tracer = lt.simulations.RayTracer(lens_type=lt.simulations.DensityPlane)
    ps = []
    for i in range(10):
        snapshot = FastPMSnapshot.open('snapshot_lensing64%d'%i)
        p,resolution,NumPart = snapshot.cutPlaneGaussianGrid(normal=2,
                                                         plane_resolution=plane_size,   
                                                         center=(1900 - i*box_size)*snapshot.Mpc_over_h,
                                                         thickness=box_size*snapshot.Mpc_over_h,
                                                         left_corner=np.zeros(3)*snapshot.Mpc_over_h,
                                                         smooth=None,
                                                         kind='density')
        ps.append(p)
        plane = DensityPlane(gaussian_filter(p,1), angle=snapshot.header["box_size"],
                                redshift=snapshot.header["redshift"],
                                cosmology=Planck15,
                                num_particles=NumPart)
        tracer.addLens(plane)
   
    tracer.addLens(lt.simulations.DensityPlane(np.zeros((128,128)),
                         angle=snapshot.header["box_size"], 
                                               redshift=1.5,
                                               cosmology=Planck15))
    # Make sure lenses are in the right order
    tracer.reorderLenses()
    return  ps, tracer
#%%
def FlowPM_raytracer(nc,plane_size,box_size,field,nsteps):
    """ Computes a convergence map using ray-tracing through an N-body for a given
    set of cosmological parameters
    """
    # Instantiates a cosmology with desired parameters
    cosmology = flowpm.cosmology.Planck15()
    r = tf.linspace(0., 2000, nsteps)
    r_center = 0.5*(r[1:] + r[:-1])
    a = flowpm.tfbackground.a_of_chi(cosmology, r)
    a_center =flowpm.tfbackground.a_of_chi(cosmology, r_center)
    init_stages = tf.linspace(0.1, a[-1], 4)
    # And initial conditions
    initial_conditions = flowpm.linear_field([nc, nc, 10 * nc],
                                             [box_size, box_size,
                                             10 * box_size], 
                                             lambda k: tf.cast(linear_matter_power(cosmology, k), tf.complex64),         
                                             batch_size=1)
    state = flowpm.lpt_init(cosmology, initial_conditions,0.1)
    med_state = flowpm.nbody(cosmology, state, init_stages,[nc, nc, 10 * nc]) 
    a_cen = np.concatenate([a_center.numpy(),np.array([a[-1]])])
    final_state, lps_a, lps, snaps = lightcone(cosmology, med_state,
                                             a_cen[::-1],
                                             [nc, nc, 10 * nc],
                                             field * 60. / plane_size,
                                             plane_size,save_snapshots=True)
    for i,(scale_factor, snap, comov_dist) in enumerate(zip(lps_a, snaps, r_center[::-1])):
      save_state(cosmology, snap, scale_factor, [nc, nc, nc], [box_size]*3, 'snapshot_lensing64%d'%i,
                 attrs={'comoving_distance': comov_dist})
    al = tf.stack(lps_a)
    z_lens=(1. / al) - 1. 
    rl = rad_comoving_distance(cosmology, al) 
    constant_factor = 3 / 2 * cosmology.Omega_m * (constants.H0 / constants.c)**2
    slice_width = box_size # Mpc/h
    density_normalization = slice_width * rl / al
    # This is the factor to apply to our lensplanes to make them LensTools compatible
    factor = density_normalization*constant_factor
    xgrid, ygrid = np.meshgrid(np.linspace(0,field,plane_size, endpoint=False), # range of X coordinates
                        np.linspace(0,field,plane_size, endpoint=False)) # range of Y coordinates
    coords = np.stack([xgrid, ygrid], axis=0)*u.deg
    c = coords.reshape([2, -1,1]).T
    # Now normalizing densityplane
    density=[]
    interp_im=[]
    for ind in range (10):
        dens = lps[ind][0]
        Npart = tf.reduce_sum(dens)
        dens = dens / Npart
        npix = plane_size
        dens =  dens * box_size**3 / ( (box_size/npix) * (box_size/npix) *(box_size/1) )
        im = tf.reshape(gaussian_filter(dens*factor[ind],4), [1, plane_size,plane_size,1])
        c1 = c.to(u.rad).value*rl[ind] / (box_size/npix)
        interp_im.append(tfa.image.interpolate_bilinear(im, c1))
        density.append(dens)
        
     
    # Exporting snapshots
    return density, interp_im, rl, z_lens,  coords, factor 

#%%
def test_densityplans():
    """ This function tests the lightcone implementation in TensorFlow 
    comparing it with Lenstools
      """
    plane_size=64
    density, interp_im, rl, z_lens,  coords, factor= FlowPM_raytracer(nc,plane_size,box_size,field,nsteps)
    ps, tracer =lenstools_raytracer(plane_size,box_size,field)
    assert_allclose(gaussian_filter(ps[0],1),gaussian_filter(density[0] * factor[0],1),rtol=1e-1)  
    
#%%

def test_interpolation():
    """ This function tests the density plan interpolation computed in our code
    comparing it with Lenstools
      """
    #Here we are considering a plane_size=1024

    data=np.load('fake_lensplane.npy')
    snapshot = FastPMSnapshot.open('snapshot_lensing_fake')
    rl=snapshot.header['comoving_distance'].value
    tracer = lt.simulations.RayTracer(lens_type=lt.simulations.DensityPlane)
    tracer.addLens(lt.simulations.DensityPlane(gaussian_filter(data,2),
                      angle=snapshot.header["box_size"], 
                                            redshift=1.5,
                                            cosmology=Planck15))
    xgrid, ygrid = np.meshgrid(np.linspace(0,field,plane_size, endpoint=False), # range of X coordinates
                        np.linspace(0,field,plane_size, endpoint=False)) # range of Y coordinates
    coords = np.stack([xgrid, ygrid], axis=0)*u.deg
    c = coords.reshape([2, -1,1]).T
    c = c.to(u.rad).value*rl / (200./plane_size)
    im = tf.reshape(gaussian_filter(data,2), [1, plane_size,plane_size,1])
    interp_im = tfa.image.interpolate_bilinear(im , c)
  
    assert_allclose(tracer.lens[0].getValues(coords[0], coords[1]),interp_im[0,:,0].numpy().reshape([plane_size,plane_size]).T, rtol=1e-1)

#%%

def test_Born_app():
    """ This function tests our Born approximation implementation 
    comparing it with Lenstools. Computes the convergence directly integrating
    the lensing density along the line of sight
      """

    density, interp_im, rl, z_lens,  coords, factor= FlowPM_raytracer(nc,plane_size,box_size,field,nsteps)
    ps, tracer =lenstools_raytracer(plane_size,box_size,field)
    lensfit=tracer.convergenceBorn(coords,z)
    kmap_flow=convergenceBorn(interp_im,rl,coords,z,plane_size)
    assert_allclose(lensfit,kmap_flow, rtol=1e-1)
#%%
