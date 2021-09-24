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

#source pypeline.sh --no_shell
eval "$(conda shell.bash hook)"
conda activate new_pypeline

which python
python -V
pip show pypeline
echo
pwd
hostname
echo

env | grep SLURM

#EO: numexpr: check env and tidy up this
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# List of environment variables set via Jenkins
echo TEST_ARCH = ${TEST_ARCH}
echo TEST_ALGO = ${TEST_ALGO}
echo TEST_DIR  = ${TEST_DIR}
OUTPUT_DIR=${TEST_DIR:-.}     # default to cwd when ENV[TEST_DIR] not set
echo OUTPUT_DIR = $OUTPUT_DIR

# Script to be run
PY_SCRIPT="./benchmarking/generic_synthesizer.py"
echo "PY_SCRIPT = $PY_SCRIPT"

# Note: --outdir is omitted, no output is written on disk

# Timing
echo "Timing - off"
time python $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO} --outdir $OUTPUT_DIR
echo; echo

# cProfile
echo "cProfile - on"
python -m cProfile -o $OUTPUT_DIR/cProfile.out $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
echo; echo

exit 0

# Nvprof
nvprof -o $OUTPUT_DIR/nvvp.out python $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
echo; echo

# Intel VTune Amplifier
amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -result-dir=$OUTPUT_DIR -- ~/miniconda3/envs/pypeline/bin/python $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
echo; echo

ls -rtl $OUTPUT_DIR
