#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=200
#SBATCH --gpus=1
#SBATCH --exclusive -A m1759
#SBATCH -J fnnestrc
#SBATCH -o ./log_slurm/%x.o%j

module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243


suffix="_test2"
niter=50
offset=True
istd=True
nx=1
ny=1

fpath=/global/cscratch1/sd/chmodi/flowpm/recon/nx${nx}_ny${ny}_mesh/fnn${suffix}/
mkdir -p $fpath
time srun python -u fnnrecon_est.py --fpath=$fpath --niter=$niter --offset=$offset --istd=$istd --nx=$nx --ny=$ny   > $fpath/log

