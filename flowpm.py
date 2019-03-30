import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

#from background import *
import tfpm 
import tfpmfuncs as tfpf
from tfpmconfig import Config




def pm(config, verbose=True):
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        linear = tfpm.linfield(config, name='linear')
        icstate = tfpm.lptinit(linear, config, name='icstate')
        fnstate = tfpm.nbody(icstate, config, verbose=verbose, name='fnstate')
        final = tf.zeros_like(linear)
        final = tfpf.cic_paint(final, fnstate[0], boxsize=bs, name='final')
        tf.add_to_collection('pm', [linear, icstate, fnstate, final])
    return g


if __name__ == "__main__":
    

    pkfile = './Planck15_a1p00.txt'
    stages = numpy.linspace(0.1, 1.0, 5, endpoint=True)
    config = Config(bs=400, nc=128, seed=100, pkfile=pkfile, stages=stages)
    bs, nc = config['boxsize'], config['nc']
    
    g = pm(config)

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        linear = g.get_tensor_by_name('linear:0')
        final = g.get_tensor_by_name('final:0')
        fnstate = g.get_tensor_by_name('fnstate:0')
        icstate = g.get_tensor_by_name('icstate:0')
        linmesh, finmesh, fstate, istate = sess.run([linear, final, fnstate, icstate])

    print(fstate[0])
    print(linmesh)
    print(finmesh)


