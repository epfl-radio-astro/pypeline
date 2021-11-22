#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --cpus-per-task 1

set -e

# TEST_ARCH defaults to --cpu if not set
[ -z $TEST_ARCH ] && TEST_ARCH="--cpu"

if   [ $TEST_ARCH == '--cpu' ]; then ARCH=CPU;
elif [ $TEST_ARCH == '--gpu' ]; then ARCH=GPU;
else 
    echo "Error: unknown TEST_ARCH option \"$TEST_ARCH\" passed."
    echo "       Only \"--cpu\" and \"--gpu\" are valid."
    exit 1
fi
echo "TEST_ARCH = $TEST_ARCH"

module load gcc
if [ $ARCH == "GPU" ]; then
    module load cuda/11.1.1;
fi
CONDA_ENV=pype-111
module list

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
conda env list

# nsys requires full path to Python interpreter
PYTHON=`which python`
echo PYTHON = $PYTHON
python -V
pip show pypeline

echo; pwd
echo; hostname
echo


# Should be safe (checked with threadpoolctl via slurm)
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1

# || true to avoid failure when grep returns nothing under set -e
echo;
env | grep UM_THREADS || true
echo
env | grep SLURM || true
echo;

# Cupy
export CUPY_CACHE_SAVE_CUDA_SOURCE=1
export CUPY_CUDA_COMPILE_WITH_DEBUG=1

# List of environment variables set via Jenkins
echo "TEST_ALGO   = ${TEST_ALGO}"
echo "TEST_BENCH  = ${TEST_BENCH}" 
echo "TEST_DIR    = ${TEST_DIR}"
echo "TEST_TRANGE = ${TEST_TRANGE}"
[ ! -z $TEST_TRANGE ] && TEST_TRANGE="--t_range ${TEST_TRANGE}"
echo "TEST_TRANGE = ${TEST_TRANGE}"
echo "TEST_SEFF   = ${TEST_SEFF}"
echo
echo "PROFILE_VTUNE    = ${PROFILE_VTUNE}"
echo "PROFILE_CPROFILE = ${PROFILE_CPROFILE}"
echo "PROFILE_NSIGHT   = ${PROFILE_NSIGHT}"

# Set profiling switches
RUN_VTUNE="${PROFILE_VTUNE:-0}"
RUN_CPROFILE="${PROFILE_CPROFILE:-0}"
RUN_NSIGHT="${PROFILE_NSIGHT:-0}"

echo $RUN_VTUNE $RUN_CPROFILE $RUN_NSIGHT


# Output directory must be defined and existing
if [[ -z "${TEST_DIR}" ]]; then
    echo "Error: TEST_DIR unset. Must point to an existing directory."
    exit 1
else 
    if [[ ! -d "${TEST_DIR}" ]]; then
        echo "Error: TEST_DIR must point to an existing directory."
        exit 1
    fi
fi
ARG_TEST_DIR="--outdir ${TEST_DIR}"


# Script to be run
PY_SCRIPT="./benchmarking/generic_synthesizer.py"
echo "PY_SCRIPT = $PY_SCRIPT"
echo; echo


echo "### Timing/memory usage"
time python $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO} ${TEST_BENCH} ${TEST_TRANGE} $ARG_TEST_DIR
ls -rtl $TEST_DIR
echo; echo


# Running with TEST_SEFF=1 causes an early exit
if [[ ! -z "${TEST_SEFF}" && ${TEST_SEFF} == "1" ]]; then
    echo "TEST_SEFF set to 1 -> exit 0";
    exit 0
fi


if [[ $ARCH == 'GPU' && $RUN_NSIGHT == "1" ]]; then
    echo "### Nsight"
    nsys --version
    nsys profile -t cuda,nvtx,osrt,cublas --sample=cpu --cudabacktrace=true --force-overwrite=true --stats=true --output=$TEST_DIR/nsys_out $PYTHON $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
    echo; echo
fi


if [ $RUN_CPROFILE == "1" ]; then
    echo "### cProfile"
    python -m cProfile -o $TEST_DIR/cProfile.out $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
    echo; echo
fi


if [ $RUN_VTUNE == "1" ]; then
    echo "### Intel VTune Amplifier"
    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh || echo "ignoring warning"
    which vtune
    echo listing of $TEST_DIR
    ls -rtl $TEST_DIR
    vtune -collect hotspots           -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_hs  -- $PYTHON $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
    vtune -collect memory-consumption -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_mem -- $PYTHON $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
    echo; echo
fi

ls -rtl $TEST_DIR
