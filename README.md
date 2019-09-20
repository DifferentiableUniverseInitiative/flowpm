# flowpm [![Build Status](https://travis-ci.org/EiffL/flowpm.svg?branch=master)](https://travis-ci.org/EiffL/flowpm)
Particle Mesh Simulation in TensorFlow, based on [fastpm-python](https://github.com/rainwoodman/fastpm-python) simulations

To install:
```
$ pip install git@github.com:modichirag/flowpm.git
```

Minimal working example is in flowpm.py. The steps are as follows:
```python
import tensorflow as tf
import flowpm

stages = np.linspace(0.1, 1.0, 10, endpoint=True)

initial_conditions = flowpm.linear_field(32,          # size of the cube
                                         100,         # Physical size of the cube
                                         ipklin,      # Initial powerspectrum
                                         batch_size=16)

# Sample particles
state = flowpm.lpt_init(initial_conditions, a0=0.1)   

# Evolve particles down to z=0
final_state = flowpm.nbody(state, stages, 32)         

# Retrieve final density field
final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])

with tf.Session() as sess:
    sim = sess.run(final_field)
```

example_graphs.py has some more graphs showing how to define a graph that does a PM simulation from an initial field, how to combine the pm graph with other modules etc.
