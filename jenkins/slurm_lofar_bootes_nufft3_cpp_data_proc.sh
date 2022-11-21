#!/bin/bash
#SBATCH --partition build
#SBATCH --time 00-01:00:00
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

PYTHON=`which python`

# Python script to be run
PY_SCRIPT="$SCRIPT_DIR/lofar_bootes_nufft3_cpp_data_proc.py"
[ -f $PY_SCRIPT ] || (echo "Fatal: PY_SCRIPT >>$PY_SCRIPT<< not found"; exit 1)
echo "PY_SCRIPT = $PY_SCRIPT"; echo

bb_print_env
bb_print_jenkins_env

# Running with TEST_SEFF=1 causes an early exit
EARLY_EXIT="${TEST_SEFF:-0}"

# Set profiling switches
RUN_CPROFILE="${PROFILE_CPROFILE:-0}"
RUN_NSIGHT="${PROFILE_NSIGHT:-0}"
RUN_VTUNE="${PROFILE_VTUNE:-0}"
RUN_ADVISOR="${PROFILE_ADVISOR:-0}"

ARG_TEST_DIR="$(bb_check_output_dir ${TEST_DIR})"


# Note: --outdir is omitted, no output is written on disk
echo "### Timing"
time python $PY_SCRIPT $ARG_TEST_DIR
echo; echo

if [[ $EARLY_EXIT == "1" ]]; then
    echo "TEST_SEFF set to 1 -> exit 0";
    exit 0
fi

if [ $RUN_CPROFILE == "1" ]; then
    echo "### cProfile"
    python -m cProfile -o $TEST_DIR/cProfile.out $PY_SCRIPT
    echo; echo
fi

if [ $RUN_VTUNE == "1" ]; then
    bb_vtune_hotspots           "${TEST_DIR}" "${PYTHON}" "{$PY_SCRIPT}"
    bb_vtune_hpc_performance    "${TEST_DIR}" "${PYTHON}" "{$PY_SCRIPT}"
    bb_vtune_memory_consumption "${TEST_DIR}" "${PYTHON}" "{$PY_SCRIPT}"
    echo; echo
fi

if [ $RUN_ADVISOR == "1" ]; then
    bb_advisor_roofline "${TEST_DIR}" "${PYTHON}" "{$PY_SCRIPT}"
fi




# To test from command line
# -------------------------

# gpu partition if no profiling
#
#export BLUEBILD_GPU=CUDA; export TMPOUT=/scratch/izar/orliac/test_nufft3_cpp; mkdir -pv $TMPOUT; PROFILE_NSIGHT=0 PROFILE_VTUNE=0 PROFILE_CPROFILE=0 TEST_SEFF=0 TEST_DIR=$TMPOUT CUPY_PYFFS=0 srun --partition gpu --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1  ./jenkins/slurm_lofar_bootes_nufft3_cpp_data_proc.sh

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/SKA/epfl-radio-astro/finufft/lib:~/SKA/epfl-radio-astro/cufinufft/lib; export FINUFFT_ROOT=~/SKA/epfl-radio-astro/finufft; export CUFINUFFT_ROOT=~/SKA/epfl-radio-astro/cufinufft; export BLUEBILD_GPU=CUDA; export TMPOUT=/scratch/izar/orliac/test_nufft3_cpp; mkdir -pv $TMPOUT; PROFILE_NSIGHT=0 PROFILE_VTUNE=0 PROFILE_CPROFILE=0 TEST_SEFF=0 TEST_DIR=$TMPOUT CUPY_PYFFS=0 srun --partition gpu --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1  ./jenkins/slurm_lofar_bootes_nufft3_cpp_data_proc.sh
# debug partion if profiling enabled
#
#export TMPOUT=/scratch/izar/orliac/test_nufft3_cpp; mkdir -pv $TMPOUT; PROFILE_NSIGHT=0 PROFILE_VTUNE=1 PROFILE_CPROFILE=1 TEST_SEFF=0 TEST_DIR=$TMPOUT CUPY_PYFFS=0 srun --partition debug --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1  ./jenkins/slurm_lofar_bootes_nufft3_cpp_data_proc.sh
