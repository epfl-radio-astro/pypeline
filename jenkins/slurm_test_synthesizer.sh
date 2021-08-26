#!/bin/bash

#SBATCH --partition build
#SBATCH --time 00-00:15:00
#SBATCH --qos gpu
#SBATCH --gres gpu:1
#SBATCH --mem 40G

set -e

module load gcc/8.4.0-cuda
module load cuda/10.2.89
module load intel-vtune
module list

source pypeline.sh --no_shell
which python
python -V

hostname

# Benchmarking
time python "./benchmarking/test_synthesizer.py"

# Profiling
amplxe-cl -collect hotspots -strategy ldconfig:notrace:notrace -- ~/miniconda3/envs/pypeline/bin/python benchmarking/test_synthesizer.py
