import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import mesh_tensorflow as mtf



def float_to_mtf(x, mesh, scalar, dtype=tf.float32):
    return mtf.import_tf_tensor(mesh, tf.constant(x, dtype=dtype, shape=[1]), shape=[scalar])



def gauss(kfield, R, kx, ky, kz, nc, bs):
    kx = tf.reshape(kx, [-1, 1, 1]) *nc/bs
    ky = tf.reshape(ky, [1, -1, 1]) *nc/bs 
    kz = tf.reshape(kz, [1, 1, -1]) *nc/bs
    kk = tf.sqrt(kx**2 + ky**2 + kz**2)
    wts = tf.exp(-0.5 * R**2 * kk**2)
    return kfield * tf.cast(wts, kfield.dtype)



def decic(kfield, kx, ky, kz, nc, bs):
    kx = tf.reshape(kx, [-1, 1, 1]) *nc/bs 
    ky = tf.reshape(ky, [1, -1, 1]) *nc/bs 
    kz = tf.reshape(kz, [1, 1, -1]) *nc/bs 
    wts = sinc(kx*bs/(2*np.pi*nc)) *sinc(ky*bs/(2*np.pi*nc)) *sinc(kz*bs/(2*np.pi*nc))
    wts = tf.pow(wts, -2.)
    return kfield * tf.cast(wts, kfield.dtype)



def fingauss(kfield, R, kx, ky, kz, nc, bs):
    kny = 1*np.pi*nc/bs
    kx = tf.reshape(kx, [-1, 1, 1]) *nc/bs 
    ky = tf.reshape(ky, [1, -1, 1]) *nc/bs 
    kz = tf.reshape(kz, [1, 1, -1]) *nc/bs 
    kk = tf.sqrt((2*kny/np.pi*tf.sin(kx*np.pi/(2*kny)))**2 + (2*kny/np.pi*tf.sin(ky*np.pi/(2*kny)))**2 + (2*kny/np.pi*tf.sin(kz*np.pi/(2*kny)))**2)
    wts = tf.exp(-0.5 * R**2 * kk**2)
    return kfield * tf.cast(wts, kfield.dtype)




def shearwts(kfield, kx, ky, kz, nc, bs):
    kx = tf.reshape(kx, [-1, 1, 1]) *nc/bs 
    ky = tf.reshape(ky, [1, -1, 1]) *nc/bs 
    kz = tf.reshape(kz, [1, 1, -1]) *nc/bs
    kv = [kx, ky, kz]
    kk = (kx**2 + ky**2 + kz**2)
    kk = tf.where(tf.abs(kk) < 1e-10, tf.ones_like(kk), kk)
    wtss = []
    for i in range(3):
        for j in range(i, 3):
            wts = tf.cast(kv[i]*kv[j] /kk, kfield.dtype)            
            if i == j: wts = wts - tf.cast(1/3., wts.dtype)
            wtss.append(kfield * wts)
    return wtss[0], wtss[1], wtss[2], wtss[3], wtss[4], wtss[5]
    


