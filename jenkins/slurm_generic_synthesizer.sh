#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G
#SBATCH --cpus-per-task 1

set -e

module load gcc/8.4.0-cuda
module load cuda/11.1.1
module list

#new_pype       + module load intel-vtune (Intel(R) VTune(TM) Amplifier 2019 Update 6 (build 602217) Command Line Tool) => FAILURE
#new_pype       + source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh  => FAILURE
#new_pype_intel + module load intel-vtune (Intel(R) VTune(TM) Amplifier 2019 Update 6 (build 602217) Command Line Tool) => FAILURE
#new_pype_intel + source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh  => SUCCESS


eval "$(conda shell.bash hook)"
CONDA_ENV=new_pype
CONDA_ENV_INTEL=new_pype_intel
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

#EO: OMP_NUM_THREADS set to 1 for M-P

# Should be safe (checked with threadpoolctl via slurm)
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1

# || true to avoid failure when grep returns nothing under set -e
echo; echo
env | grep UM_THREADS || true
echo
env | grep SLURM || true
echo; echo

# Cupy
export CUPY_CACHE_SAVE_CUDA_SOURCE=1
export CUPY_CUDA_COMPILE_WITH_DEBUG=1

# List of environment variables set via Jenkins
echo "TEST_ARCH   = ${TEST_ARCH}"
echo "TEST_ALGO   = ${TEST_ALGO}"
echo "TEST_BENCH  = ${TEST_BENCH}" 
echo "TEST_DIR    = ${TEST_DIR}"
echo "TEST_TRANGE = ${TEST_TRANGE}"
[ -z $TEST_ARCH ] && TEST_ARCH="--cpu" #avoids no def
echo "TEST_ARCH   = ${TEST_ARCH}"
[ ! -z $TEST_TRANGE ] && TEST_TRANGE="--t_range ${TEST_TRANGE}"
echo "TEST_TRANGE = ${TEST_TRANGE}"
OUTPUT_DIR=${TEST_DIR:-.}     # default to cwd when ENV[TEST_DIR] not set
echo OUTPUT_DIR = $OUTPUT_DIR

# Script to be run
PY_SCRIPT="./benchmarking/generic_synthesizer.py"
echo "PY_SCRIPT = $PY_SCRIPT"

# Note: --outdir is omitted, no output is written on disk

echo "Timing"
time python $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO} ${TEST_BENCH} ${TEST_TRANGE} --outdir $OUTPUT_DIR
ls -rtl $OUTPUT_DIR
echo; echo

#exit 0

echo "Nsight"
nsys --version
nsys profile -t cuda,nvtx,osrt,cublas --sample=cpu --cudabacktrace=true --force-overwrite=true --stats=true --output=nsys_out $PYTHON $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
echo; echo

#exit 0

echo "cProfile"
python -m cProfile -o $OUTPUT_DIR/cProfile.out $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
echo; echo

#echo "nvprof"
#nvprof -o $OUTPUT_DIR/nvvp.out python $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
#echo; echo

# Intel VTune Amplifier (CPU only, don't have permissions for GPU)
if [ $TEST_ARCH != '--gpu' ]; then
    echo "Intel VTune Amplifier"

    # Warning: for running vtune we need intel environment!
    if [ $CONDA_ENV != $CONDA_ENV_INTEL ]; then 
        conda deactivate
        conda activate $CONDA_ENV_INTEL
        PYTHON=`which python`
    fi
    echo PYTHON = $PYTHON
    $PYTHON -V

    ##which amplxe-cl
    ##amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -result-dir=$OUTPUT_DIR -- $PYTHON $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
    source /work/scitas-ge/richart/test_stacks/syrah/v1/opt/spack/linux-rhel7-skylake_avx512/gcc-8.4.0/intel-oneapi-vtune-2021.6.0-34ym22fgautykbgmg5hhgkiwrvbwfvko/setvars.sh || echo "ignoring warning"
    which vtune
    vtune -collect hotspots -run-pass-thru=--no-altstack -strategy ldconfig:notrace:notrace -search-dir=. -result-dir=$OUTPUT_DIR -- $PYTHON $PY_SCRIPT ${TEST_ARCH} ${TEST_ALGO}
else
    echo "Lack of permissions to run Intel VTune Amplifier on GPU hardware. To be investigated."
fi
echo; echo

ls -rtl $OUTPUT_DIR
