#!/bin/bash
#SBATCH --nodes=4
#SBATCH -q gpu_preempt
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=30
#SBATCH --gres=gpu:8
#SBATCH --exclusive -A m1759
 
module purge && module load  esslurm gcc/7.3.0 python3 cuda/10.1.243



#for nc in 128 256 512 1024 2048; do 
for nc in 2048 1024 512 256; do 
    nx=32
    ny=1
    echo
    echo Starting $nx $ny
    if [[ $(find ./timelogs/log_$nc-b1$nx-b2$ny -type f -size +1000c 2>/dev/null) ]]
       then
	   echo ./timelogs/log_$nc-b1$nx-b2$ny exists
    else
	time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:$nx;b2:$ny" --layout="nx:b1;ny:b2" --output_file="timeline-b1$nx-b2$ny" > ./timelogs/log_$nc-b1$nx-b2$ny
    fi

    nx=16
    ny=2
    echo
    echo Starting $nx $ny
    if [[ $(find ./timelogs/log_$nc-b1$nx-b2$ny -type f -size +1000c 2>/dev/null) ]]
       then
	   echo ./timelogs/log_$nc-b1$nx-b2$ny exists
    else
	time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:$nx;b2:$ny" --layout="nx:b1;ny:b2" --output_file="timeline-b1$nx-b2$ny" > ./timelogs/log_$nc-b1$nx-b2$ny
    fi

    nx=8
    ny=4
    echo
    echo Starting $nx $ny
    if [[ $(find ./timelogs/log_$nc-b1$nx-b2$ny -type f -size +1000c 2>/dev/null) ]]
       then
	   echo ./timelogs/log_$nc-b1$nx-b2$ny exists
    else
	time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:$nx;b2:$ny" --layout="nx:b1;ny:b2" --output_file="timeline-b1$nx-b2$ny" > ./timelogs/log_$nc-b1$nx-b2$ny
    fi


    nx=4
    ny=8
    echo
    echo Starting $nx $ny
    if [[ $(find ./timelogs/log_$nc-b1$nx-b2$ny -type f -size +1000c 2>/dev/null) ]]
       then
	   echo ./timelogs/log_$nc-b1$nx-b2$ny exists
    else
	time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:$nx;b2:$ny" --layout="nx:b1;ny:b2" --output_file="timeline-b1$nx-b2$ny" > ./timelogs/log_$nc-b1$nx-b2$ny
    fi


    nx=2
    ny=16
    echo
    echo Starting $nx $ny
    if [[ $(find ./timelogs/log_$nc-b1$nx-b2$ny -type f -size +1000c 2>/dev/null) ]]
       then
	   echo ./timelogs/log_$nc-b1$nx-b2$ny exists
    else
	time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:$nx;b2:$ny" --layout="nx:b1;ny:b2" --output_file="timeline-b1$nx-b2$ny" > ./timelogs/log_$nc-b1$nx-b2$ny
    fi



    nx=1
    ny=32
    echo
    echo Starting $nx $ny
    if [[ $(find ./timelogs/log_$nc-b1$nx-b2$ny -type f -size +1000c 2>/dev/null) ]]
       then
	   echo ./timelogs/log_$nc-b1$nx-b2$ny exists
    else
	time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:$nx;b2:$ny" --layout="nx:b1;ny:b2" --output_file="timeline-b1$nx-b2$ny" > ./timelogs/log_$nc-b1$nx-b2$ny
    fi


done
##
##
##    echo
##    
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:16;b2:2" --layout="nx:b1;ny:b2" --output_file="timeline-b116-b22" > ./timelogs/log_$nc-b116-b22
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:2;b2:16" --layout="nx:b1;ny:b2" --output_file="timeline-b12-b28" > ./timelogs/log_$nc-b12-b216
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:8;b2:4" --layout="nx:b1;ny:b2" --output_file="timeline-b18-b24" > ./timelogs/log_$nc-b18-b24
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:4;b2:8" --layout="nx:b1;ny:b2" --output_file="timeline-b14-b28" > ./timelogs/log_$nc-b114-b28;
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:1;b2:32" --layout="nx:b1;ny:b2" --output_file="timeline-b11-b232" > ./timelogs/log_$nc-b11-b232;
##done

##
##for nc in 1024; do 
##    echo
##    ##pip install --user -e ../.
##    echo
##    echo Starting
##    echo
##    #time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:32;b2:1" --layout="nx:b1;ny:b2" --output_file="timeline-b132-b21" > ./timelogs/log_$nc-b132-b21
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:16;b2:2" --layout="nx:b1;ny:b2" --output_file="timeline-b116-b22" > ./timelogs/log_$nc-b116-b22
##    echo
##    echo Starting
##    echo
##    #time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:2;b2:16" --layout="nx:b1;ny:b2" --output_file="timeline-b12-b28" > ./timelogs/log_$nc-b12-b216
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:8;b2:4" --layout="nx:b1;ny:b2" --output_file="timeline-b18-b24" > ./timelogs/log_$nc-b18-b24
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:4;b2:8" --layout="nx:b1;ny:b2" --output_file="timeline-b14-b28" > ./timelogs/log_$nc-b114-b28;
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:1;b2:32" --layout="nx:b1;ny:b2" --output_file="timeline-b11-b232" > ./timelogs/log_$nc-b11-b232;
##done
##
##
##for nc in 2048 ; do 
##    echo
##    ##pip install --user -e ../.
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:32;b2:1" --layout="nx:b1;ny:b2" --output_file="timeline-b132-b21" > ./timelogs/log_$nc-b132-b21
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:16;b2:2" --layout="nx:b1;ny:b2" --output_file="timeline-b116-b22" > ./timelogs/log_$nc-b116-b22
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:2;b2:16" --layout="nx:b1;ny:b2" --output_file="timeline-b12-b28" > ./timelogs/log_$nc-b12-b216
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:8;b2:4" --layout="nx:b1;ny:b2" --output_file="timeline-b18-b24" > ./timelogs/log_$nc-b18-b24
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:4;b2:8" --layout="nx:b1;ny:b2" --output_file="timeline-b14-b28" > ./timelogs/log_$nc-b114-b28;
##    echo
##    echo Starting
##    echo
##    time srun python fft_benchmark.py --cube_size=$nc --batch_size=1 --mesh_shape="b1:1;b2:32" --layout="nx:b1;ny:b2" --output_file="timeline-b11-b232" > ./timelogs/log_$nc-b11-b232;
##done
##
##
