# flowpm [![Build Status](https://travis-ci.org/modichirag/flowpm.svg?branch=master)](https://travis-ci.org/modichirag/flowpm)[![PyPI version](https://badge.fury.io/py/flowpm.svg)](https://badge.fury.io/py/flowpm)
Particle Mesh Simulation in TensorFlow, based on [fastpm-python](https://github.com/rainwoodman/fastpm-python) simulations

To install:
```
$ pip install flowpm
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


## Mesh Tensorflow implementation

First install like so:
```bash
$ pip install --user mesh-tensorflow
$ git clone -b u/EiffL/mesh git@github.com:modichirag/flowpm.git
$ cd flowpm
$ pip install --user -e .
```

Go to the example folder:
```bash
$ cd examples
```

Then start the mesh servers, e.g. on a single machine with 2 GPUs:
```bash
$ CUDA_VISIBLE_DEVICES=0; python mesh_lpt.py --mesh_hosts=localhost:2222,localhost:2223 --job_name=mesh --task_index=0
```

```bash
$ CUDA_VISIBLE_DEVICES=1; python mesh_lpt.py --mesh_hosts=localhost:2222,localhost:2223 --job_name=mesh --task_index=1
```

And finally run the code:
```bash
$ CUDA_VISIBLE_DEVICES=2; python mesh_lpt.py --mesh_hosts=localhost:2222,localhost:2223 --job_name=main --nc=128
```

## Cori-specific instruction

Starting from a clean environment, these are the steps to execute on cori-gpu nodes. Info on using the Cori GPU nodes here: https://docs-dev.nersc.gov/cgpu/

0) Login to a cori-gpu node to prepare the environment:
```
$ module add esslurm
$ salloc -C gpu -N 1 -t 30 -c 10 --gres=gpu:1 -A m1759
```

1) First compile and install mpi4py
```
$ module purge && module load tensorflow/gpu-1.15.0-rc1-py37 esslurm gcc/7.3.0 cuda mvapich2
$ wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz
$ tar zxvf mpi4py-3.0.0.tar.gz
$ cd mpi4py-3.0.0
$ python setup.py build --mpicc=mpicc
$ python setup.py install --user
```

2) Install dependencies
```
$ pip install --user mesh-tensorflow
```

3) Install the Mesh TensorFlow branch of FlowPM
```
$ git clone https://github.com/modichirag/flowpm.git
$ cd flowpm
$ git checkout u/EiffL/mesh
$ pip install --user -e .
```

4) To run the demo comparing the distributed computation to single GPU:
```
$ cd examples
$ sbatch lpt_job.sh
```

This will generate a plot `comparison.png` showing from a set of initial
conditions, the result of a single LPT step on single GPU TensorFlow vs Mesh
TensorFlow.
