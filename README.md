# FlowPM
[![Build Status](https://travis-ci.org/DifferentiableUniverseInitiative/flowpm.svg?branch=master)](https://travis-ci.org/DifferentiableUniverseInitiative/flowpm) [![PyPI version](https://badge.fury.io/py/flowpm.svg)](https://badge.fury.io/py/flowpm) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DifferentiableUniverseInitiative/flowpm/blob/master/notebooks/flowpm_tutorial.ipynb) [![arXiv:2010.11847](https://img.shields.io/badge/astro--ph.IM-arXiv%3A2010.11847-B31B1B.svg)](https://arxiv.org/abs/2010.11847) [![youtube](https://img.shields.io/badge/-youtube-red?logo=youtube&labelColor=grey)](https://youtu.be/DHOaHTU61hM)   [![yapf](https://img.shields.io/badge/code%20style-yapf-blue.svg)](https://www.python.org/dev/peps/pep-0008/) [![Documentation Status](https://readthedocs.org/projects/flowpm/badge/?version=latest)](https://flowpm.readthedocs.io/en/documentation/?badge=documentation)


Particle Mesh Simulation in TensorFlow, based on [fastpm-python](https://github.com/rainwoodman/fastpm-python) simulations

Try me out: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DifferentiableUniverseInitiative/flowpm/blob/master/notebooks/flowpm_blog.ipynb)

To install:
```
$ pip install flowpm
```

For a minimal working example of FlowPM, see this [notebook](notebooks/flowpm_demo.ipynb). The steps are as follows:
```python
import tensorflow as tf
import numpy as np
import flowpm

cosmo = flowpm.cosmology.Planck15()
stages = np.linspace(0.1, 1.0, 10, endpoint=True)

initial_conditions = flowpm.linear_field(32,          # size of the cube
                                         100,         # Physical size of the cube
                                         ipklin,      # Initial power spectrum
                                         batch_size=16)

# Sample particles
state = flowpm.lpt_init(cosmo, initial_conditions, a0=0.1)   

# Evolve particles down to z=0
final_state = flowpm.nbody(cosmo, state, stages, 32)         

# Retrieve final density field
final_field = flowpm.cic_paint(tf.zeros_like(initial_conditions), final_state[0])
```

## Mesh TensorFlow implementation

FlowPM provides a [Mesh TensorFlow](https://github.com/tensorflow/mesh) implementation of FastPM, 
for running distributed simulations across very large supercomputers. 

### Instructions for GPU clusters

We rely on a customized Mesh TensorFlow backend based on [Horovod](https://github.com/horovod/horovod) to
distribute computations on GPU clusters through the high performance [NCCL library](https://developer.nvidia.com/nccl).

To install the necessary dependencies, you first need to be in an environment providing:
  - TensorFlow 2.1 or above
  - NCCL 2.8 or above
You can then install Horovod and Mesh TensorFlow with: 
```bash
$ pip install git+https://github.com/horovod/horovod.git
$ pip install git+https://github.com/DifferentiableUniverseInitiative/mesh@hvd_max
```

### TPU setup

To run FlowPM on Google TPUs here is the procedure

 - Step 1: Setting up a cloud TPU in the desired zone, do from the GCP console:
 ```
$ gcloud config set compute/region europe-west4
$ gcloud config set compute/zone europe-west4-a
$ ctpu up --name=flowpm --tpu-size=v3-32
 ```

  - Step 2: Installing dependencies and FlowPM:
```
$ git clone https://github.com/DifferentiableUniverseInitiative/flowpm.git
$ cd flowpm
$ git checkout mesh
$ pip3 install --user mesh-tensorflow
$ pip3 install --user -e .
```

It's so easy, it's almost criminal.

#### Notes on using and profiling for TPUs

There a few things to keep in mind when using TPUs, in particular, the section
on `Excessive tensor padding` from this document: https://cloud.google.com/tpu/docs/troubleshooting

See the [README](scripts/README.md) in the script folder for more info on how to profile
