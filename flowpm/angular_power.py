#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:56:46 2021

@author: Denise Lanzieri, Benjamin Remy

"""
import numpy as np

#%%
def radial_profile(data):
    """
    Compute the radial profile of 2d image
    :param data: 2d image
    :return: radial profile
    """
    center = data.shape[0]/2
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r = r.astype('int32')

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def measure_power_spectrum(map_data,field,nc_xy):
    """
    measures power 2d data
    :param map_data: map (n x n)
    :param pixel_size: pixel size (rad/pixel)
    :return: ell
    :return: power spectrum
    """
    data_ft = np.fft.fftshift(np.fft.fft2(map_data)) / map_data.shape[0]
    nyquist = np.int(map_data.shape[0]/2)
    power_spectrum = radial_profile(np.real(data_ft*np.conj(data_ft)))[:nyquist]
    power_spectrum = power_spectrum*pixel_size(field,nc_xy)**2

    k = np.arange(power_spectrum.shape[0])
    ell = 2. * np.pi * k / pixel_size(field,nc_xy) / map_data.shape[0]


    return ell, power_spectrum


def resolution(field,nc_xy):
    """
    pixel resolution
    
    Parameters:
    -----------
    field: int or float
        transveres degres of the field
    
    nc_xy : int
       Number of cell for x and  y 
        
    Returns
    -------
      float
     pixel resolution
     
    """
    return  field*60/nc_xy



def pixel_size(field,nc_xy):
    """
    pixel size
    
    Parameters:
    -----------
    field: int or float
        transveres degres of the field
    
    nc_xy : int
       Number of cell for x and  y 
        
    Returns
    -------
    
    pizel size: float
    pixel size
    
    Notes
    -----

    The pixels size is given by:

    .. math::

        pixel_size =  =pi * resolution / 180. / 60. #rad/pixel
     
    """
    return field/nc_xy / 180 *np.pi

