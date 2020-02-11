#!/bin/bash
#SBATCH --nodes=1
#SBATCH -q gpu_preempt
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=200
#SBATCH --gres=gpu:8
#SBATCH --exclusive -A m1759
#SBATCH -J recon
#SBATCH -o ./log_slurm/%x.o%j

module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243

hsize=16

nc=64
nx=4
ny=2


mkdir -p /global/cscratch1/sd/chmodi/flowpm/recon/nbody_${nc}_nx${nx}_ny${ny}
time srun python -u pyramid_recon.py --nbody=True --gpus_per_task=8 --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize  > /global/cscratch1/sd/chmodi/flowpm/recon/nbody_${nc}_nx${nx}_ny${ny}/log

#mkdir -p /global/cscratch1/sd/chmodi/flowpm/recon/lpt_${nc}_nx${nx}_ny${ny}
#time srun python -u pyramid_recon.py --nbody=False --gpus_per_task=8 --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize  > /global/cscratch1/sd/chmodi/flowpm/recon/lpt_${nc}_nx${nx}_ny${ny}/log
