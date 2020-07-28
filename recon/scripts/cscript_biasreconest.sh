#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=200
#SBATCH --gpus=1
#SBATCH --exclusive -A m1759
#SBATCH -D /global/homes/c/chmodi/Programs/flowpm-recon/recon
#SBATCH -J biasestrc
#SBATCH -o ./log_slurm/%x.o%j

module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243


suffix="_test"
niter=50
nx=1
ny=1
nc=256

fpath=/global/cscratch1/sd/chmodi/flowpm/recon/nx${nx}_ny${ny}_mesh/bias${suffix}/
mkdir -p $fpath
time srun python -u biasrecon_est.py --fpath=$fpath --niter=$niter --nx=$nx --ny=$ny --nc=$nc    > $fpath/log

