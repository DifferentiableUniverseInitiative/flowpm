import numpy as np
import numpy
import os, sys

import tensorflow as tf
import tensorflow_hub as hub

import tfpm
import tfpmfuncs as tfpf
from tfpmconfig import Config

def graphpm(config, verbose=True, initlin=False):
    '''return graph to do pm simulation
    if initlin is False, the returned graph generates initial conditions
    if initlin is True, the returned graph has a placeholder'''
    bs, nc = config['boxsize'], config['nc']
    g = tf.Graph()
    with g.as_default():

        linmesh = tf.placeholder(tf.float32, (nc, nc, nc), name='linmesh')
        if initlin:
            linear = tf.Variable(0.)
            linear = tf.assign(linear, linmesh, validate_shape=False, name='linear')
        else:
            linear = tfpm.linfield(config, name='linear')
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=verbose, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=config['boxsize'], name='final')
        tf.add_to_collection('pm', [linear, icstate, fnstate, final])
    return g



def genpm(config, linmesh=None, verbose=True):
    '''do pm sim to generate final matter field'''
    ##Generate Data
    bs, nc = config['boxsize'], config['nc']
    if linmesh is None:
        g = graphpm(config, initlin=False)
        linmesh = np.zeros((nc, nc, nc))
    else:
        g = graphpm(config, initlin=True, verbose=verbose)

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())
        linmesh_t = g.get_tensor_by_name('linmesh:0')
        linear_t = g.get_tensor_by_name('linear:0')
        final_t = g.get_tensor_by_name('final:0')
        linear, final = session.run([linear_t, final_t], {linmesh_t:linmesh})

    return linear, final



#Example graphs to pass in the final field into a module of trained network
#Use this as template to modify based on signature of module
def graphlintomodel(config, modpath, pad=False, ny=1):
    '''return graph to do pm sim and then sample halo positions from it'''
    bs, nc = config['boxsize'], config['nc']

    g = tf.Graph()
    with g.as_default():
        module = hub.Module(modpath)

        linmesh = tf.placeholder(tf.float32, (nc, nc, nc), name='linmesh')
        datamesh = tf.placeholder(tf.float32, (nc, nc, nc, ny), name='datamesh')

        #PM
        linear = tf.Variable(0.)
        linear = tf.assign(linear, linmesh, validate_shape=False, name='linear')
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=False, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
        #Sample
        if pad:
            xx = tf.concat((final[-pad:, :, :], final, final[:pad, :, :]), axis=0)
            xx = tf.concat((xx[:, -pad:, :], xx, xx[:, :pad, :]), axis=1)
            xx = tf.concat((xx[:, :, -pad:], xx, xx[:, :, :pad]), axis=2)
            xx = tf.expand_dims(tf.expand_dims(xx, 0), -1)
        else: xx = tf.assign(final)

        yy = tf.expand_dims(datamesh, 0)
        samples = module(dict(features=xx, labels=yy), as_dict=True)['sample']
        samples = tf.identity(samples, name='samples')
        loglik = module(dict(features=xx, labels=yy), as_dict=True)['loglikelihood']
        loglik = tf.identity(loglik, name='loglik')

        tf.add_to_collection('inits', [linmesh, datamesh])
        tf.add_to_collection('reconpm', [linear, final, fnstate, samples, loglik])

    return g

def genlintomodel(config, modpath, linmesh, datamesh, pad=False):
    '''do pm sim to generate final matter field'''
    ##Generate Data
    bs, nc = config['boxsize'], config['nc']
    ny = datamesh.shape[-1]
    g = graphlintomodel(config, modpath, pad=pad, ny=ny)
    print('\nGraph constructed\n')

    with tf.Session(graph=g) as session:
        session.run(tf.global_variables_initializer())
        linmesh_t = g.get_tensor_by_name('linmesh:0')
        datamesh_t = g.get_tensor_by_name('datamesh:0')
        linear_t = g.get_tensor_by_name('linear:0')
        final_t = g.get_tensor_by_name('final:0')
        samples_t = g.get_tensor_by_name('samples:0')
        loglik_t = g.get_tensor_by_name('loglik:0')
        linear, final, data, loglik = session.run([linear_t, final_t, samples_t,loglik_t],
                                             {linmesh_t:linmesh, datamesh_t:datamesh})

    return linear, final, data
