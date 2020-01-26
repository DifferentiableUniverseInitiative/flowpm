#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=10
#SBATCH --gres=gpu:2
#SBATCH --exclusive -A m1759
module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243
srun python pyramid_nbody_SLURM.py --nc=128 > log-play
