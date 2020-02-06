#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=10
#SBATCH --gres=gpu:8
#SBATCH --exclusive -A m1759
module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243

hsize=16
##for nc in 256 512 1024 ; do 
##    echo $nc
##    echo
##    nx=1
##    ny=16
##    echo Starting $nx $ny
##    echo
##    time srun python pyramid_lpt_benchmark.py --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize --output_file="./timelines/timeline_pyramid_$nc-b1$nx-b2$ny"  > ./timelogs/log_pyramidlpt_$nc-b1$nx-b2$ny
##
##

nc=64
nx=2
ny=1
time srun python numgradients.py --gpus_per_task=2 --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize  > log_grads_$nc-b1$nx-b2$ny
