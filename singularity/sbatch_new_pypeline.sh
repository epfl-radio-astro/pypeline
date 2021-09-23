#!/bin/bash

#SBATCH --time 00-00:15:00
#SBATCH --partition build #gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G

set -e

module load gcc
module load cuda
module list

export SINGULARITY_BINDPATH=$CUDA_ROOT
export SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIBRARY

singularity run --nv -B /work /work/scitas-share/SKA/singularity/new_pypeline.sif python ~/SKA/pypeline/benchmarking/test_fastsynthesizer.py
