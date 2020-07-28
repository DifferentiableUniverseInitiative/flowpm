#!/bin/bash
#SBATCH --nodes=1
###SBATCH -q gpu_preempt
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=30
#SBATCH --gpus=1
#SBATCH --exclusive -A m1759
#SBATCH -J meshnbrecsgd0p02
#SBATCH -o ./log_slurm/%x.o%j

module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243

hsize=16

nx=1
ny=1
gpus_per_task=1
#suffix="_paper_adam0p01_100"
suffix="_paper_sgd_0p02anneal"

#nc=64
#fpath=/global/cscratch1/sd/chmodi/flowpm/recon/lpt_${nc}_nx${nx}_ny${ny}_mesh${suffix}
#mkdir -p $fpath
#time srun python -u mesh_recon.py --suffix=$suffix --nbody=False --gpus_per_task=$gpus_per_task --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize  > $fpath/log


for nc in 128; do
    echo
    echo $nc
    echo
    fpath=/global/cscratch1/sd/chmodi/flowpm/recon/nbody_${nc}_nx${nx}_ny${ny}_mesh${suffix}
    mkdir -p $fpath
    time srun python -u mesh_reconest.py --suffix=$suffix --nbody=True --gpus_per_task=$gpus_per_task --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize  > $fpath/log
done
