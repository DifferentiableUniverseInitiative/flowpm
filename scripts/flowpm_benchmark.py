import tensorflow as tf
import numpy as np
import flowpm

from flowpm.tfpower import linear_matter_power
from flowpm.tfbackground import cosmo

tf.flags.DEFINE_integer("nc", 128, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 100, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")

FLAGS = tf.flags.FLAGS

@tf.function
def simulation(om, s8):
    cosmo['sigma8'] = tf.convert_to_tensor(s8, dtype=tf.float32)
    cosmo['Omega0_m'] = tf.convert_to_tensor(om, dtype=tf.float32)
    
    stages = np.linspace(FLAGS.a0, FLAGS.af, FLAGS.nsteps, endpoint=True) #time-steps for the integration
    
    initial_conditions = flowpm.linear_field(FLAGS.nc,                # size of the cube
                                             FLAGS.box_size,          # Physical size of the cube
                                             lambda k: tf.cast(linear_matter_power(cosmo, k), tf.complex64), # Initial powerspectrum
                                             batch_size=FLAGS.batch_size)

    # Sample particles
    state = flowpm.lpt_init(initial_conditions, FLAGS.a0)   

    # Evolve particles down to z=0
    final_state = flowpm.nbody(state, stages, FLAGS.nc)         

    # Retrieve final density field
    final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])
    
    return final_field

# run the simulation 10 times
for i in range(10):
    final_field = simulation(0.3075, 0.8159)
    res = final_field.numpy()

