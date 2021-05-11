#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:20:00
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3

module purge && module load cgpu esslurm tensorflow/2.4.1-gpu

export SLURM_CPU_BIND="cores"

srun python mesh_nbody_benchmark.py --nc=512 --batch_size=1 --nx=2 --ny=2 --hsize=32