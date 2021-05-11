#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=15
#SBATCH --gres=gpu:1
#SBATCH --exclusive -A m1759
module purge && module load esslurm gcc/7.3.0 python3 cuda/10.1.243

srun nvprof -f -o test.nvvp python fft_benchmark-nvprof.py --cube_size=512 --batch_size=2 --mesh_shape="b1:2" --gpus_per_node=1 --gpus_per_task=1 --layout="nx:b1,tny:b1" > log_1024
