# Scripts used for benchmarking the implementation


## Benchmarking on Cori

To run these scripts on Cori-GPU:

  - *make sure* you have setup the environment as described in the home [README](../README.md)
  for Cori specific instructions.

  - make sure you are loading the `esslurm` module before you can actually submit these scripts
  ```
  $ module add esslurm
  ```

### Benchmarking distributed FFTs implementation

TL;DR, run from this folder:
```
$ sbatch fft_nvprof.sh
```
This uses `nvprof` to record the execution of a series of 512^3 FFTs distributed
across 2 GPUs, on 2 nodes. The trace of this execution is record as an nvprof file
which can be read using the `nvvp` utility.

To change the configuration for running this test, the `fft_nvprof.sh` script to
be modified in 2 places:
  - The *number of nodes* and *number of gpus* per nodes in the header
  - The following parameters in the python call: `--mesh_shape`, `--gpus_per_node` `--gpus_per_task`

Here is an example to run the same size transform but on two GPUs on the same node:
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --time=15
#SBATCH --gres=gpu:2
#SBATCH --exclusive -A m1759
module purge && module load esslurm gcc/7.3.0 python3 cuda/10.1.243

srun nvprof -f -o test.nvvp python fft_benchmark-nvprof.py --cube_size=512 --batch_size=2 --mesh_shape="b1:2" --gpus_per_node=2 --gpus_per_task=2 --layout="nx:b1,tny:b1" > log_1024
```

### Benchmarking the full simulation
1) Run from this folder
```
$ module add esslurm
$ sbatch cori_benchmark_job.sh
```
And have a look at the log file


## Benchmarking on TPUs

To run a benchmark on TPUs here are the steps to follow

0) Make sure your cloud VM environment is correctly setup, see the README TPU section

1) From a first VM shell, start the benchmarking script:
```
$ cd flowpm/scripts
$ python3 fft_benchmark_TPU.py --model_dir=gs://flowpm_eu/tpu_test
```

2) From a second VM shell, start Tensorboard as such:
```
$ export PATH="$PATH:`python3 -m site --user-base`/bin"
$ export STORAGE_BUCKET=gs://flowpm_eu/tpu_profiling/run0
$ export MODEL_DIR=gs://flowpm_eu/tpu_test
$ tensorboard --logdir=${MODEL_DIR} &
```
Finally, click the web preview button on the top right corner to launch TensorBoard

3) To capture the TPU trace, go to the profile tool, use these settings:
TPU name: flowpm
Address Type: TPU Name
Profiling duration: 10000

4) When you have acquired the profile, you can shutdown the running benchmarking
script with `ctrl+\`

Alternatively, one can also save the trace and then visualize it in tensorboard. To do so, 

1) Follow step 1) above and start the benchmarking script

2) From a second VM shell that has again been setup to start the TPU as pointed in main README, do the following:
```
$ export PATH="$PATH:`python3 -m site --user-base`/bin"
$ export TPU_NAME=flowpm
$ export MODEL_DIR=gs://flowpm_eu/tpu_test
$ capture_tpu_profile --tpu=$TPU_NAME --logdir=${MODEL_DIR} --duration_ms=10000 --num_tracing_attempts=10
```
Ideally this will end with some output like - _Profile session succeed for host(s):10.240.1.5,10.240.1.2,10.240.1.4,10.240.1.3_

3) From a third VM shell, follow the step 2) above to launch tensorboard and visualize


More info on using TensorBoard and Profiling for TPU here:
  - https://cloud.google.com/tpu/docs/tensorboard-setup
  - https://cloud.google.com/tpu/docs/cloud-tpu-tools

To access these traces from your local computer, here are the 2 simple steps
  - Use the gcloud cli to authenticate yourself:
  ```
  $ gcloud auth application-default login
  ```
  - Start TensorBoard with the path to your Bucket:
  ```
  $ tensorboard --logdir=gs://flowpm_eu/tpu_test
  ```
  - Step 3: Profit!
