# Scripts used for benchmarking the implementation


## Benchmarking on Cori

To run this script on Cori-GPU

0) Make sure you have setup the environment as described in the home README for Cori specific instructions

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
