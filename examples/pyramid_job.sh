#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=10
#SBATCH --gres=gpu:2
#SBATCH --exclusive -A m1759
module purge && module load  tensorflow/gpu-2.0.0-py37 esslurm gcc/7.3.0 cuda
srun python pyramid_nbody_SLURM.py --nc=128 > log-play
