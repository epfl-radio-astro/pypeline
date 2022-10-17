#!/bin/bash
#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --cpus-per-task 1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -e

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
source $SCRIPT_DIR/install.sh
bb_load_gcc_stack
bb_activate_venv
bb_pip_install 'cupy-cuda110'

PYTHON=$(which python)

# Python script to be run
PY_SCRIPT="$SCRIPT_DIR/generic_synthesizer.py"
[ -f $PY_SCRIPT ] || (echo "Fatal: PY_SCRIPT >>$PY_SCRIPT<< not found"; exit 1)
echo "PY_SCRIPT = $PY_SCRIPT"; echo

bb_print_env
bb_print_jenkins_env

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

# Cupy
export CUPY_CACHE_SAVE_CUDA_SOURCE=1
export CUPY_CUDA_COMPILE_WITH_DEBUG=1

[ ! -z $TEST_TRANGE ] && TEST_TRANGE="--t_range ${TEST_TRANGE}"

# Set early exit switch
EARLY_EXIT="${TEST_SEFF:-0}"

# Set profiling switches
RUN_CPROFILE="${PROFILE_CPROFILE:-0}"
RUN_NSIGHT="${PROFILE_NSIGHT:-0}"
RUN_VTUNE="${PROFILE_VTUNE:-0}"
RUN_ADVISOR="${PROFILE_ADVISOR:-0}"


ARG_TEST_DIR="$(bb_check_output_dir ${TEST_DIR})"

echo "### Timing/memory usage"
time python $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO} ${TEST_BENCH} ${TEST_TRANGE} $ARG_TEST_DIR
ls -rtl $TEST_DIR
echo; echo

# Running with TEST_SEFF=1 causes an early exit
if [ $EARLY_EXIT == "1" ]; then
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
    bb_vtune_hotspots           "${TEST_DIR}" "${PYTHON}" "{$PY_SCRIPT}" "${TEST_ARCH}" "${TEST_ALGO}"
    bb_vtune_hpc_performance    "${TEST_DIR}" "${PYTHON}" "{$PY_SCRIPT}" "${TEST_ARCH}" "${TEST_ALGO}"
    bb_vtune_memory_consumption "${TEST_DIR}" "${PYTHON}" "{$PY_SCRIPT}" "${TEST_ARCH}" "${TEST_ALGO}"
    echo; echo
fi



# To test from command line
#export TMPOUT=/scratch/izar/orliac/test_pype-111b/; mkdir -pv $TMPOUT; PROFILE_NSIGHT=1 PROFILE_VTUNE=1 PROFILE_CPROFILE=1 TEST_TRANGE=5 TEST_SEFF=0 TEST_DIR=$TMPOUT srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1  ./jenkins/slurm_generic_synthesizer.sh
