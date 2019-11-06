#!/bin/bash
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=5
#SBATCH --gres=gpu:8
#SBATCH --exclusive -A m1759
module purge && module load tensorflow/gpu-1.15.0-rc1-py37 esslurm gcc/7.3.0 cuda mvapich2

srun python fft_benchmark.py --cube_size=1024 --batch_size=1 > log_1024
