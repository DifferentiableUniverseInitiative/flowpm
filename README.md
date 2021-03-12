# flowpm
[![Build Status](https://travis-ci.org/DifferentiableUniverseInitiative/flowpm.svg?branch=master)](https://travis-ci.org/DifferentiableUniverseInitiative/flowpm) [![PyPI version](https://badge.fury.io/py/flowpm.svg)](https://badge.fury.io/py/flowpm) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DifferentiableUniverseInitiative/flowpm/blob/master/notebooks/flowpm_tutorial.ipynb) [![arXiv:2010.11847](https://img.shields.io/badge/astro--ph.IM-arXiv%3A2010.11847-B31B1B.svg)](https://arxiv.org/abs/2010.11847) [![youtube](https://img.shields.io/badge/-youtube-red?logo=youtube&labelColor=grey)](https://youtu.be/DHOaHTU61hM)   [![PEP8](https://img.shields.io/badge/code%20style-pep8-blue.svg)](https://www.python.org/dev/peps/pep-0008/)


Particle Mesh Simulation in TensorFlow, based on [fastpm-python](https://github.com/rainwoodman/fastpm-python) simulations

Try me out: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DifferentiableUniverseInitiative/flowpm/blob/master/notebooks/flowpm_blog.ipynb)

To install:
```
$ pip install flowpm
```

For a minimal working example of FlowPM, see this [notebook](notebook/flowpm_tutorial.ipynb). The steps are as follows:
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

FlowPM provides a Mesh TensorFlow implementation of FastPM, for running distributed
simulations across very large supercomputers.

Here are the instructions for installing and running on Cori-GPU. More info about
this machine here: https://docs-dev.nersc.gov/cgpu/

0) Login to a cori-gpu node to prepare the environment:
```
$ module add esslurm
$ salloc -C gpu -N 1 -t 30 -c 10 --gres=gpu:1 -A m1759
```

1) First install dependencies
```
$ module purge && module load gcc/7.3.0 python3 cuda/10.1.243
$ pip install --user tensorflow==2.1
$ pip install --user mesh-tensorflow
```
**NOTE: we are installing our own tensorflow 2.1 version until a module is available at NERSC**

3) Install the Mesh TensorFlow branch of FlowPM
```
$ git clone https://github.com/DifferentiableUniverseInitiative/flowpm.git
$ cd flowpm
$ git checkout mesh
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
