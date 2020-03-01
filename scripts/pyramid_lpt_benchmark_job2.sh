#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=30
#SBATCH --gres=gpu:8
#SBATCH --exclusive -A m1759

module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243


hsize=32
#for nc in 128, 256 512 1024 ; do 
for nc in 128 ; do 
    echo $nc
    echo
    nx=1
    ny=16
    echo Starting $nx $ny
    echo
    time srun python pyramid_lpt_benchmark.py --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize --output_file="./timelines/timeline_pyramid_$nc-b1$nx-b2$ny"  > ./lpt-timelogs/log_pyramidlpt_$nc-b1$nx-b2$ny
    echo
    nx=2
    ny=8
    echo Starting $nx $ny
    echo
    time srun python pyramid_lpt_benchmark.py --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize --output_file="./timelines/timeline_pyramid_$nc-b1$nx-b2$ny"  > ./lpt-timelogs/log_pyramidlpt_$nc-b1$nx-b2$ny
    echo
    nx=4
    ny=4
    echo Starting $nx $ny
    echo
    time srun python pyramid_lpt_benchmark.py --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize --output_file="./timelines/timeline_pyramid_$nc-b1$nx-b2$ny"  > ./lpt-timelogs/log_pyramidlpt_$nc-b1$nx-b2$ny
    echo
    nx=8
    ny=2
    echo Starting $nx $ny
    echo
    time srun python pyramid_lpt_benchmark.py --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize --output_file="./timelines/timeline_pyramid_$nc-b1$nx-b2$ny"  > ./lpt-timelogs/log_pyramidlpt_$nc-b1$nx-b2$ny
    echo
    nx=16
    ny=1
    echo Starting $nx $ny
    echo
    time srun python pyramid_lpt_benchmark.py --nc=$nc --batch_size=1 --mesh_shape="row:$nx;col:$ny" --nx=$nx --ny=$ny --hsize=$hsize --output_file="./timelines/timeline_pyramid_$nc-b1$nx-b2$ny"  > ./lpt-timelogs/log_pyramidlpt_$nc-b1$nx-b2$ny

done

