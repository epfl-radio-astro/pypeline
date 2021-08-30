#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --cpus-per-task 4

set -e

module load gcc/8.4.0-cuda
module load cuda/10.2.89
module load intel-vtune
module list

source pypeline.sh --no_shell
which python
python -V
pip show pypeline

hostname

echo "UTC_TAG = $UTC_TAG"

exit 0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK


echo 

# Debug
#time python "./benchmarking/debug.py"
#amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -- ~/miniconda3/envs/pypeline/bin/python ./benchmarking/debug.py

# Timing
echo;echo;echo
#time python "./benchmarking/test_synthesizer.py"
time python "./benchmarking/test_synthesizer_cpu.py"

# cProfile
time python -m cProfile -o cpu.pstats "./benchmarking/test_synthesizer_cpu.py"

exit 0

# Profiling
echo;echo;echo
amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -- ~/miniconda3/envs/pypeline/bin/python benchmarking/test_synthesizer.py
amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -- ~/miniconda3/envs/pypeline/bin/python benchmarking/test_synthesizer_cpu.py
