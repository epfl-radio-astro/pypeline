#!/bin/sh

module load gcc fftw cuda/11.0

export FINUFFT_ROOT=/home/orliac/SKA/epfl-radio-astro/finufft/
export PYTHONPATH=$FINUFFT_ROOT/python/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FINUFFT_ROOT/lib

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rascil38

#pip list

python make_skalow_ms070.py

python skalow_test_nufft3_eo.py
