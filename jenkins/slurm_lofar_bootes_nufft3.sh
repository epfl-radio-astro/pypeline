#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-01:00:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --cpus-per-task 1

set -e

module load gcc
#module load cuda/10
module load fftw
#CONDA_ENV=pynuf102-dbg
CONDA_ENV=pype-111
module list

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
conda env list

which python
python -V
pip show pypeline
echo
pwd
hostname
echo

# nsys requires full path to Python interpreter
PYTHON=`which python`
echo PYTHON = $PYTHON

#FINUFFT=./finufft
#cd $FINUFFT
#FINUFFT_DIR=`pwd` python -m pip install -e ./python
$PYTHON -c "import finufft as _; print(_.__path__)" # Print path to finufft
#cd ..
#exit 0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# || true to avoid failure when grep returns nothing under set -e
echo; echo
env | grep UM_THREADS || true
echo
env | grep SLURM || true
echo; echo

# List of environment variables set via Jenkins
echo "TEST_DIR    = ${TEST_DIR}"
OUTPUT_DIR=${TEST_DIR:-.}     # default to cwd when ENV[TEST_DIR] not set
echo OUTPUT_DIR = $OUTPUT_DIR

# Script to be run
PY_SCRIPT="./examples/simulation/lofar_bootes_nufft3.py"
echo "PY_SCRIPT = $PY_SCRIPT"; echo


# Note: --outdir is omitted, no output is written on disk
echo "Timing"
time python $PY_SCRIPT --outdir $OUTPUT_DIR
ls -rtl $OUTPUT_DIR
echo; echo

echo "EARLY EXIT for faster tests"
exit 0

# Intel VTune Amplifier (CPU only, don't have permissions for GPU)
if [ 1 == 1 ]; then
    echo "Intel VTune Amplifier"

    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh || echo "ignoring warning"
    which vtune
    echo listing of $OUTPUT_DIR
    ls -rtl $OUTPUT_DIR
    vtune -collect hotspots -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -search-dir=. -result-dir=$OUTPUT_DIR/vtune -- $PYTHON $PY_SCRIPT
#else
#    echo "Lack of permissions to run Intel VTune Amplifier on GPU hardware. To be investigated."
fi
echo; echo


# Disable Advisor as really slow (> 1 hour) and pointless if we do
# not care about the CPU version of finufft
if [ 1 == 0 ]; then
    echo "Advisor"
    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-advisor-2021.4.0-any7cfov5s4ujprr7plf7ks7xzoyqljz/setvars.sh
    ADVIXE_RUNTOOL_OPTIONS=--no-altstack OMP_NUM_THREADS=1 advixe-cl -collect roofline --enable-cache-simulation --profile-python -project-dir $OUTPUT_DIR/advisor -search-dir src:=. -- $PYTHON $PY_SCRIPT
fi

ls -rtl $OUTPUT_DIR
