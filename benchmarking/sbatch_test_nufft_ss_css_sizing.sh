#!/bin/bash

#SBATCH --mem 80G
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --partition debug
#SBATCH --cpus-per-task 20
#SBATCH --time 00-00:05:00

set -e

module load gcc
module load cuda/11
module load fftw
module load intel-mkl
module list

eval "$(conda shell.bash hook)"
conda activate pype-111
which python
python -V

export MKL_VERBOSE=0
export OMP_DISPLAY_AFFINITY=0
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PY_SCRIPT=./test_nufft_ss_css_sizing.py
ls -l $PY_SCRIPT

python $PY_SCRIPT
