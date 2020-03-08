#!/bin/bash
#SBATCH --nodes=1
#SBATCH -q gpu_preempt
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=200
#SBATCH --gres=gpu:1
#SBATCH --exclusive -A m1759
#SBATCH -J fnnestrc
#SBATCH -o ./log_slurm/%x.o%j

module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243


suffix="_off"
niter=100
offset=True
istd=True

fpath=/global/cscratch1/sd/chmodi/flowpm/recon/nx${nx}_ny${ny}_mesh/fnn${suffix}/
mkdir -p $fpath
time srun python -u fnnrecon_est.py --fpath=$fpath --niter=$niter --offset=$offset --istd=$istd   > $fpath/log

