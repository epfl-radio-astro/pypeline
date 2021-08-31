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
echo
hostname
echo

#EO: numexpr: check env and tidy up this
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

PY_SCRIPT="./benchmarking/test_synthesizer_cpu.py"

# Timing
time python $PY_SCRIPT

# cProfile
time python -m cProfile -o $TEST_DIR/cProfile.out $PY_SCRIPT

exit 0

# Profiling
echo;echo;echo
#amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -- ~/miniconda3/envs/pypeline/bin/python benchmarking/test_synthesizer.py
amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -- ~/miniconda3/envs/pypeline/bin/python benchmarking/test_synthesizer_cpu.py
