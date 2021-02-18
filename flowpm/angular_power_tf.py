#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:54:33 2021

@author: Denise Lanzieri, Benjamin Remy
"""

import tensorflow as tf
import numpy as np


def measure_power_spectrum_tf(map_data,field,nc_xy):
    """
    Measures power spectrum or 2d data
    
    Parameters:
    -----------
    map_data: map (n x n)
    
    field: int or float
        transveres degres of the field
        
    nc_xy : int
           Number of pixel for x and  y 
          
    Returns
    -------
    ell: tf.TensorArray
    power spectrum: tf.TensorArray
    """
    
    def radial_profile_tf(data):
        """
        Compute the radial profile of 2d image
        
        Parameters:
        -----------
        data: 2d image
        
        Returns
        -------
        radial profile
        """
        center = data.shape[0]/2
        y, x = np.indices((data.shape))
        r = tf.math.sqrt((x - center)**2 + (y - center)**2)
        r=tf.cast(r,dtype=tf.int32)
        tbin=tf.math.bincount(tf.reshape(r,[-1]), tf.reshape(data,[-1]))
        nr = tf.math.bincount(tf.reshape(r,[-1]))
        radialprofile=tf.cast(tbin,dtype=tf.float64)/tf.cast(nr,dtype=tf.float64)
        return radialprofile
    
    
    def resolution(field,nc_xy):
        """
        pixel resolution

        Returns
        -------
          float
         pixel resolution
         
        """
        return  field*60/nc_xy
    
    def pixel_size_tf(field,nc_xy):
        """
        pixel size

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
    data_ft = tf.signal.fftshift(tf.signal.fft2d(map_data)) / map_data.shape[0]
    nyquist = tf.cast(map_data.shape[0]/2,dtype=tf.int32)
    power_spectrum = radial_profile_tf(tf.math.real(data_ft*tf.math.conj(data_ft)))[:nyquist]
    power_spectrum = power_spectrum*pixel_size_tf(field,nc_xy)**2
    k = tf.range(power_spectrum.shape[0],dtype=tf.float64)
    ell = 2. * tf.constant(np.pi,dtype=tf.float64) * k / tf.constant(pixel_size_tf(field,nc_xy),dtype=tf.float64) / tf.cast(map_data.shape[0],dtype=tf.float64)
    return ell, power_spectrum






