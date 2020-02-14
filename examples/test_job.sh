#!/bin/bash
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=300
#SBATCH --gres=gpu:8
#SBATCH --exclusive -A m1759
#SBATCH -J test
#SBATCH -o ./log_slurm/%x.o%j
#SBATCH -q gpu_preempt

module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243

hsize=16

nx=4
ny=4
gpus_per_task=8
suffix="_test"

for nc in 256 ; do
    echo
    echo $nc
    echo
    fpath=/global/cscratch1/sd/chmodi/flowpm/recon/nbody_${nc}_nx${nx}_ny${ny}_mesh${suffix}
    mkdir -p $fpath
    time srun python -u test256recon.py  --suffix=$suffix --nbody=True --gpus_per_task=$gpus_per_task --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize  > ${fpath}/log
done


