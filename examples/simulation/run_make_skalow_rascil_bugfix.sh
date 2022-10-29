#!/bin/sh

module purge
module load gcc/9.3.0-cuda
module load mvapich2/2.3.4
module load fftw/3.3.8-mpi-openmp
module load cuda/11.0.2

source ~/SKA/epfl-radio-astro/PYPE1102/bin/activate


# BB
python skalow_test_nufft3_eo2.py
