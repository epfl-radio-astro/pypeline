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

# Python script to be run
PY_SCRIPT="$SCRIPT_DIR/lofar_bootes_nufft3.py"
[ -f $PY_SCRIPT ] || (echo "Fatal: PY_SCRIPT >>$PY_SCRIPT<< not found"; exit 1)
echo "PY_SCRIPT = $PY_SCRIPT"; echo

# || true to avoid failure when grep returns nothing under set -e
echo; echo
env | grep THREADS || true
echo
env | grep SLURM || true
echo; echo

echo "TEST_DIR    = ${TEST_DIR}"
echo "TEST_SEFF   = ${TEST_SEFF}"
echo
echo "PROFILE_CPROFILE = ${PROFILE_CPROFILE}"
echo "PROFILE_NSIGHT   = ${PROFILE_NSIGHT}"
echo "PROFILE_VTUNE    = ${PROFILE_VTUNE}"
echo "PROFILE_ADVISOR  = ${PROFILE_ADVISOR}"

# Set early exit switch
EARLY_EXIT="${TEST_SEFF:-0}"

# Set profiling switches
RUN_CPROFILE="${PROFILE_CPROFILE:-0}"
RUN_NSIGHT="${PROFILE_NSIGHT:-0}"
RUN_VTUNE="${PROFILE_VTUNE:-0}"
RUN_ADVISOR="${PROFILE_ADVISOR:-0}"

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


# Note: --outdir is omitted, no output is written on disk
echo "### Timing"
time python $PY_SCRIPT $ARG_TEST_DIR
echo; echo

# Running with TEST_SEFF=1 causes an early exit
if [ $EARLY_EXIT == "1" ]; then
    echo "TEST_SEFF set to 1 -> exit 0";
    exit 0
fi


if [ $RUN_CPROFILE == "1" ]; then
    echo "### cProfile"
    python -m cProfile -o $TEST_DIR/cProfile.out $PY_SCRIPT
    echo; echo
fi


if [ $RUN_VTUNE == "1" ]; then
    echo "### Intel VTune Amplifier"
    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh || echo "ignoring warning"
    which vtune
    vtune -collect hotspots           -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_hs  -- $PYTHON $PY_SCRIPT
    vtune -collect hpc-performance    -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_hpc -- $PYTHON $PY_SCRIPT
    vtune -collect memory-consumption -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TEST_DIR/vtune_mem -- $PYTHON $PY_SCRIPT
fi
echo; echo


if [ $RUN_ADVISOR == "1" ]; then
    echo "### Intel Advisor"
    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-advisor-2021.4.0-any7cfov5s4ujprr7plf7ks7xzoyqljz/setvars.sh
    ADVIXE_RUNTOOL_OPTIONS=--no-altstack OMP_NUM_THREADS=1 advixe-cl -collect roofline --enable-cache-simulation --profile-python -project-dir $TEST_DIR/advisor -search-dir src:=. -- $PYTHON $PY_SCRIPT
fi

ls -rtl $TEST_DIR

# To test from command line
#export PYTHONPATH=./tests/ss_cpp/build_GCC/python/; export TMPOUT=/scratch/izar/orliac/test_pype-111e/; mkdir -pv $TMPOUT; PROFILE_NSIGHT=0 PROFILE_VTUNE=1 PROFILE_CPROFILE=0 TEST_SEFF=0 TEST_DIR=$TMPOUT CUPY_PYFFS=0 srun --partition build --time 00-00:15:00 --qos gpu --gres gpu:1 --mem 40G --cpus-per-task 1  ./jenkins/slurm_lofar_bootes_nufft3.sh


# Hel + salloc + ssh
#conda activate pype-111
#module load gcc fftw
#export TMPOUT=/scratch/orliac/test_pype-111f/; mkdir -pv $TMPOUT; CUPY_PYFFS=0 vtune -collect hpc-performance -knob stack-size=0 -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -source-search-dir=. -search-dir=. -result-dir=$TMPOUT/vtune_hpc -- ~/miniconda3/envs/pype-111/bin/python ./examples/simulation/lofar_bootes_nufft3.py
